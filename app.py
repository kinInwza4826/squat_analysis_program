
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from aiortc.mediastreams import VideoFrame
from typing import Union, List, Dict, Any
import tempfile
import os
import io # Added for graph download functionality

# --- Internal Constants for Moment Arm Derivation ---
# These ratios are simplified and illustrative.
# In a real advanced biomechanical model, these would be derived from detailed anatomical data.
MOMENT_ARM_RATIOS = {
    'quad': 0.05,          # e.g., Quadriceps moment arm is 5% of shin length
    'glute': 0.10,         # e.g., Gluteal moment arm is 10% of thigh length
    'hamstring_hip': 0.08, # e.g., Hamstring moment arm at hip is 8% of thigh length
    'erector_spinae': 0.07 # e.g., Erector Spinae moment arm is 7% of torso length
}

# --- Biomechanics Calculation Functions ---
def _calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three points a, b, and c, where b is the vertex."""
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def _calculate_force(tau, r, theta):
    """Calculates muscle force based on torque, moment arm, and angle."""
    # Prevent division by zero or near-zero cosine values which can lead to extreme force estimates
    if r == 0 or math.isclose(math.cos(theta), 0, abs_tol=1e-9): # Use math.isclose for float comparison
        return 0
    
    force = tau / (r * math.cos(theta))

    # --- Clipping Force Values for better visualization ---
    # Forces should generally be positive when muscles are contracting against gravity.
    # Large negative values or excessively high positive values are often artifacts.
    # Clip to a reasonable physiological range.
    min_force = 0 # Muscles generate positive force to counteract load
    max_force = 100000 # Cap at 100 kN (100,000 N) to prevent extreme spikes

    return np.clip(force, min_force, max_force)

def _calculate_young_modulus(F, L0, A, delta_L):
    """Calculates Young's Modulus based on force, original length, area, and change in length."""
    # Prevent division by zero
    if A * delta_L == 0:
        return 0
    
    young_modulus = (F * L0) / (A * delta_L)

    # --- Clipping Young's Modulus Values for better visualization ---
    # Young's Modulus should always be positive. Cap at a reasonable upper limit.
    min_modulus = 0 # Modulus cannot be negative
    max_modulus = 5000e6 # Cap at 5000 MPa (5 GPa) to prevent extreme spikes (converted to Pa)

    return np.clip(young_modulus, min_modulus, max_modulus)

def _estimate_max_physiological_strain(age):
    """
    Estimates a general maximum physiological strain percentage for tendons based on age.
    This is a simplified model for guidance and is not a precise medical limit.
    """
    if age < 0: return 0 # Invalid age
    if age < 20: return 5.0 # Young, more elastic
    elif age < 40: return 4.0 # Adult, good elasticity
    elif age < 60: return 3.0 # Middle-aged, decreased elasticity
    else: return 2.0 # Older, less elastic

# --- Video Processor Class for Streamlit-WebRTC (for Live Camera) ---
class SquatBiomechanicsProcessor(VideoProcessorBase):
    def __init__(self, constants: Dict[str, Any]):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.constants = constants
        
        # log_data and data_to_display are managed in st.session_state in the main thread
        # The processor only needs to update st.session_state
        pass 

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image with MediaPipe Pose.
        results = self.pose.process(image_rgb)

        # Draw the pose annotation on the image.
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        data_to_display = {
            "Knee Angle": 0, "Hip Angle": 0, "Back Angle": 0,
            "Quad F": 0, "Glute F": 0, "Hamstring F": 0, "Back F": 0,
            "Strain": 0,
            "Quad E": 0, "Glute E": 0, "Hamstring E": 0, "Back E": 0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            try:
                # Get coordinates for key joints
                shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                     landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
                hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y])
                knee = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y])
                ankle = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y])
            except Exception as e:
                st.warning(f"Partial landmark detection or error: {e}")
                if 'data_to_display' in st.session_state:
                    st.session_state.data_to_display = data_to_display
                self._draw_landmarks(image_bgr, results)
                return VideoFrame.from_ndarray(image_bgr, format="bgr24")

            # Biomechanics Calculations
            knee_angle = _calculate_angle(hip, knee, ankle)
            hip_angle = _calculate_angle(shoulder, hip, knee)
            back_angle = _calculate_angle(shoulder, hip, knee) # Simplified: using same points as hip angle

            theta_knee = math.radians(180 - knee_angle)
            theta_hip = math.radians(180 - hip_angle)

            F_external = (self.constants['weight_kg'] * 9.81) / 2 
            phi_rad = self.constants['phi']
            F_parallel = F_external * math.cos(phi_rad)

            tau_knee = F_parallel * self.constants['r_external']
            tau_hip = F_parallel * self.constants['r_external']

            # --- Derived Moment Arms from Lengths ---
            r_quad_derived = self.constants['shin_length_m'] * MOMENT_ARM_RATIOS['quad']
            r_glute_derived = self.constants['thigh_length_m'] * MOMENT_ARM_RATIOS['glute']
            r_hamstring_hip_derived = self.constants['thigh_length_m'] * MOMENT_ARM_RATIOS['hamstring_hip']
            r_erector_spinae_derived = self.constants['torso_length_m'] * MOMENT_ARM_RATIOS['erector_spinae']

            F_quad = _calculate_force(tau_knee, r_quad_derived, theta_knee)
            F_glute = _calculate_force(tau_hip, r_glute_derived, theta_hip)
            F_hamstring = _calculate_force(tau_hip, r_hamstring_hip_derived, theta_hip)
            F_erector_spinae = _calculate_force(tau_hip, r_erector_spinae_derived, theta_hip)

            # Strain is now derived from age-based physiological limit
            strain_percentage_from_age = _estimate_max_physiological_strain(self.constants['age_years'])
            strain = strain_percentage_from_age / 100.0 # Convert percentage to ratio for calculations

            # L0 is still needed for Young's Modulus calculation, but delta_L_ratio is now internal
            # E_quad = _calculate_young_modulus(F_quad, self.constants['L0'], self.constants['A_quad'], delta_L)
            # Re-calculate delta_L based on the new strain value for Young's Modulus
            delta_L_for_modulus = strain * self.constants['L0']

            E_quad = _calculate_young_modulus(F_quad, self.constants['L0'], self.constants['A_quad'], delta_L_for_modulus)
            E_glute = _calculate_young_modulus(F_glute, self.constants['L0'], self.constants['A_glute'], delta_L_for_modulus)
            E_hamstring = _calculate_young_modulus(F_hamstring, self.constants['L0'], self.constants['A_hamstring'], delta_L_for_modulus)
            E_erector_spinae = _calculate_young_modulus(F_erector_spinae, self.constants['L0'], self.constants['A_erector_spinae'], delta_L_for_modulus)

            data_to_display = {
                "Knee Angle": int(knee_angle),
                "Hip Angle": int(hip_angle),
                "Back Angle": int(back_angle),
                "Quad F": int(F_quad),
                "Glute F": int(F_glute),
                "Hamstring F": int(F_hamstring),
                "Back F": int(F_erector_spinae),
                "Strain": strain * 100, # Convert to percentage for display
                "Quad E": E_quad / 1e6, # Convert to MPa
                "Glute E": E_glute / 1e6, # Convert to MPa
                "Hamstring E": E_hamstring / 1e6, # Convert to MPa
                "Back E": E_erector_spinae / 1e6 # Convert to MPa
            }

            if 'data_to_display' in st.session_state:
                st.session_state.data_to_display = data_to_display

            if st.session_state.get('analysis_running', False) and 'log_data' in st.session_state:
                timestamp = time.time()
                st.session_state.log_data.append([
                    timestamp, knee_angle, hip_angle, back_angle,
                    F_quad, F_glute, F_hamstring, F_erector_spinae,
                    strain, E_quad, E_glute, E_hamstring, E_erector_spinae
                ])

            self._draw_landmarks(image_bgr, results)

        return VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def _draw_landmarks(self, image, results):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                      self.mp_pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                     )

# --- Video File Processor Function ---
def process_video_file(uploaded_file, constants):
    mp_pose = mp.solutions.pose
    pose = mp.solutions.pose.Pose() # Initialize pose here for consistency

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        os.unlink(video_path)
        return [], []

    processed_frames = []
    log_data_file = []
    start_time = time.time()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Processing video: {uploaded_file.name} (Total frames: {frame_count}, FPS: {fps:.2f})")
    progress_bar = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) # Convert to RGB for st.image

        data_to_display = {
            "Knee Angle": 0, "Hip Angle": 0, "Back Angle": 0,
            "Quad F": 0, "Glute F": 0, "Hamstring F": 0, "Back F": 0,
            "Strain": 0,
            "Quad E": 0, "Glute E": 0, "Hamstring E": 0, "Back E": 0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
                hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
                knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y])
                ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
            except Exception:
                pass # Skip biomechanics if landmarks are incomplete

            if all(v is not None for v in [shoulder, hip, knee, ankle]): # Ensure all points are detected
                knee_angle = _calculate_angle(hip, knee, ankle)
                hip_angle = _calculate_angle(shoulder, hip, knee)
                back_angle = _calculate_angle(shoulder, hip, knee)

                theta_knee = math.radians(180 - knee_angle)
                theta_hip = math.radians(180 - hip_angle)

                F_external = (constants['weight_kg'] * 9.81) / 2 
                phi_rad = constants['phi']
                F_parallel = F_external * math.cos(phi_rad)

                tau_knee = F_parallel * constants['r_external']
                tau_hip = F_parallel * constants['r_external']

                # --- Derived Moment Arms from Lengths for Video File Processing ---
                r_quad_derived = constants['shin_length_m'] * MOMENT_ARM_RATIOS['quad']
                r_glute_derived = constants['thigh_length_m'] * MOMENT_ARM_RATIOS['glute']
                r_hamstring_hip_derived = constants['thigh_length_m'] * MOMENT_ARM_RATIOS['hamstring_hip']
                r_erector_spinae_derived = constants['torso_length_m'] * MOMENT_ARM_RATIOS['erector_spinae']

                F_quad = _calculate_force(tau_knee, r_quad_derived, theta_knee)
                F_glute = _calculate_force(tau_hip, r_glute_derived, theta_hip)
                F_hamstring = _calculate_force(tau_hip, r_hamstring_hip_derived, theta_hip)
                F_erector_spinae = _calculate_force(tau_hip, r_erector_spinae_derived, theta_hip)

                # Strain is now derived from age-based physiological limit
                strain_percentage_from_age = _estimate_max_physiological_strain(constants['age_years'])
                strain = strain_percentage_from_age / 100.0 # Convert percentage to ratio for calculations

                delta_L_for_modulus = strain * constants['L0']

                E_quad = _calculate_young_modulus(F_quad, constants['L0'], constants['A_quad'], delta_L_for_modulus)
                E_glute = _calculate_young_modulus(F_glute, constants['L0'], constants['A_glute'], delta_L_for_modulus)
                E_hamstring = _calculate_young_modulus(F_hamstring, constants['L0'], constants['A_hamstring'], delta_L_for_modulus)
                E_erector_spinae = _calculate_young_modulus(F_erector_spinae, constants['L0'], constants['A_erector_spinae'], delta_L_for_modulus)
                
                log_data_file.append([
                    time.time() - start_time, knee_angle, hip_angle, back_angle,
                    F_quad, F_glute, F_hamstring, F_erector_spinae,
                    strain, E_quad, E_glute, E_hamstring, E_erector_spinae
                ])

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(image_bgr, results.pose_landmarks,
                                                      mp_pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                     )
        
        processed_frames.append(image_bgr) # Store RGB for st.image
        frame_idx += 1
        progress_bar.progress(frame_idx / frame_count)

    cap.release()
    pose.close()
    os.unlink(video_path) # Clean up temporary file
    st.success("Video processing complete!")
    return processed_frames, log_data_file

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Real-Time Squat Biomechanics Analysis")
    st.title("🏋️ Real-Time Squat Biomechanics Analysis")

    # Initialize ALL session state variables unconditionally at the very top of main()
    if 'constants' not in st.session_state:
        st.session_state.constants = {
            'age_years': 30,
            'weight_kg': 80.0,
            'shin_length_m': 0.40, # Default average shin length
            'thigh_length_m': 0.45, # Default average thigh length
            'torso_length_m': 0.50, # Default average torso length
            'r_external': 0.35,
            'phi_deg': 15.0,
            'phi': math.radians(15.0), # Calculated from phi_deg
            # 'delta_L_ratio': 0.08, # Removed as it's now derived from age
            'A_quad': 0.00006, 'A_glute': 0.00009, 'A_hamstring': 0.00007, 'A_erector_spinae': 0.00003,
            'L0': 0.10,
        }
    if 'log_data' not in st.session_state:
        st.session_state.log_data = []
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'data_to_display' not in st.session_state:
        st.session_state.data_to_display = {
            "Knee Angle": 0, "Hip Angle": 0, "Back Angle": 0,
            "Quad F": 0, "Glute F": 0, "Hamstring F": 0, "Back F": 0,
            "Strain": 0,
            "Quad E": 0, "Glute E": 0, "Hamstring E": 0, "Back E": 0
        }
    if 'processed_video_frames' not in st.session_state:
        st.session_state.processed_video_frames = []
    if 'processed_video_log_data' not in st.session_state:
        st.session_state.processed_video_log_data = []

    # Create a local copy of constants to pass to the video processor factory.
    # IMPORTANT: Update delta_L_ratio based on age here before passing to processor
    st.session_state.constants['delta_L_ratio'] = _estimate_max_physiological_strain(st.session_state.constants['age_years']) / 100.0
    current_constants = st.session_state.constants.copy()

    # --- Sidebar for Controls and Constants ---
    with st.sidebar:
        st.header("⚙️ Settings & Controls")

        analysis_mode = st.radio(
            "Choose Analysis Mode:",
            ("Live Camera Analysis", "Upload Video for Analysis")
        )

        st.subheader("Basic User Inputs")
        st.session_state.constants['age_years'] = st.number_input(
            "Age (years):", min_value=0, max_value=150, value=st.session_state.constants['age_years'], step=1,
            help="Your age in years. Used to estimate a general physiological strain limit for tendons."
        )
        st.session_state.constants['weight_kg'] = st.number_input(
            "Weight (kg):", min_value=0.0, value=st.session_state.constants['weight_kg'], step=1.0, format="%.1f",
            help="Enter the total weight being lifted (body weight + barbell weight)."
        )
        
        st.subheader("Anatomical Lengths (for Biomechanical Model)")
        st.session_state.constants['shin_length_m'] = st.number_input(
            "Shin Length (m):", min_value=0.01, value=st.session_state.constants['shin_length_m'], step=0.01, format="%.2f",
            help="Length of your shin (tibia) from knee to ankle. Used to derive quadriceps moment arm."
        )
        st.session_state.constants['thigh_length_m'] = st.number_input(
            "Thigh Length (m):", min_value=0.01, value=st.session_state.constants['thigh_length_m'], step=0.01, format="%.2f",
            help="Length of your thigh (femur) from hip to knee. Used to derive glute and hamstring moment arms."
        )
        st.session_state.constants['torso_length_m'] = st.number_input(
            "Torso Length (m):", min_value=0.01, value=st.session_state.constants['torso_length_m'], step=0.01, format="%.2f",
            help="Length of your torso from hip to shoulder. Used to derive erector spinae moment arm."
        )

        with st.expander("Advanced Biomechanical Model Parameters"):
            st.session_state.constants['r_external'] = st.number_input(
                "External Moment Arm (m):", min_value=0.0, value=st.session_state.constants['r_external'], step=0.01, format="%.2f",
                help="Distance from the joint (hip/knee) to the line of action of the external force. Adjust based on barbell position."
            )
            st.session_state.constants['phi_deg'] = st.number_input(
                "Hip Abduction Angle (deg):", min_value=0.0, max_value=90.0, value=st.session_state.constants['phi_deg'], step=1.0, format="%.1f",
                help="Angle representing the load's line of action relative to the body's sagital plane. Helps account for non-vertical force components."
            )
            st.session_state.constants['phi'] = math.radians(st.session_state.constants['phi_deg']) # Update phi_rad
            
            # Removed delta_L_ratio input as it's now derived from age
            # st.session_state.constants['delta_L_ratio'] = st.number_input(
            #     "Delta L Ratio:", min_value=0.0, max_value=1.0, value=st.session_state.constants['delta_L_ratio'], step=0.01, format="%.2f",
            #     help="Ratio of the change in tendon length to the resting length. Directly influences the calculated strain."
            # )
            st.write(f"**Derived Delta L Ratio (from Age):** {st.session_state.constants['delta_L_ratio']:.2f}")
            st.write(f"*(This means the calculated Strain (%) will match the Estimated Max Strain for your age.)*")

            st.session_state.constants['A_quad'] = st.number_input(
                "Quad Area (m²):", min_value=0.0, value=st.session_state.constants['A_quad'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the quadriceps muscle. Larger area generally means higher force potential."
            )
            st.session_state.constants['A_glute'] = st.number_input(
                "Glute Area (m²):", min_value=0.0, value=st.session_state.constants['A_glute'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the gluteal muscles. Affects glute force and Young's Modulus."
            )
            st.session_state.constants['A_hamstring'] = st.number_input(
                "Hamstring Area (m²):", min_value=0.0, value=st.session_state.constants['A_hamstring'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the hamstring muscles. Influences hamstring force contribution."
            )
            st.session_state.constants['A_erector_spinae'] = st.number_input(
                "Erector Spinae Area (m²):", min_value=0.0, value=st.session_state.constants['A_erector_spinae'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the erector spinae muscles. Critical for back extensor force."
            )
            st.session_state.constants['L0'] = st.number_input(
                "Resting Tendon Length (m):", min_value=0.0, value=st.session_state.constants['L0'], step=0.01, format="%.2f",
                help="Initial, unstretched length of the muscle-tendon unit. Baseline for strain calculation."
            )

        st.subheader("Control Analysis")
        # Buttons for Live Camera Mode
        if analysis_mode == "Live Camera Analysis":
            col1_btn, col2_btn = st.columns(2)
            with col1_btn:
                if st.button("Start Live Analysis", key="start_live_btn"):
                    st.session_state.analysis_running = True
                    st.session_state.log_data = [] # Clear previous data on start
                    st.session_state.processed_video_frames = [] # Clear processed video data
                    st.session_state.processed_video_log_data = []
                    st.success("Live analysis started! Perform your squats.")
            with col2_btn:
                if st.button("Stop Live Analysis", key="stop_live_btn"):
                    st.session_state.analysis_running = False
                    st.info("Live analysis stopped. You can now save data.")
        # Button for Video Upload Mode
        else: # analysis_mode == "Upload Video for Analysis"
            uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])
            if uploaded_file is not None:
                if st.button("Analyze Uploaded Video", key="analyze_uploaded_video_btn"):
                    st.session_state.analysis_running = False # Ensure live analysis is off
                    st.session_state.log_data = [] # Clear live data
                    st.session_state.processed_video_frames = [] # Clear previous processed video
                    st.session_state.processed_video_log_data = []
                    
                    with st.spinner("Processing video... This may take a while depending on video length."):
                        processed_frames, processed_log_data = process_video_file(uploaded_file, current_constants)
                        st.session_state.processed_video_frames = processed_frames
                        st.session_state.processed_video_log_data = processed_log_data
                    st.success("Video analysis complete! See results below.")
            else:
                st.write("Upload a video to begin analysis.")


        st.subheader("Save Options")
        # Determine which log data to use for saving/graphing
        data_to_save = []
        if analysis_mode == "Live Camera Analysis" and st.session_state.log_data:
            data_to_save = st.session_state.log_data
        elif analysis_mode == "Upload Video for Analysis" and st.session_state.processed_video_log_data:
            data_to_save = st.session_state.processed_video_log_data

        if data_to_save:
            df_log = pd.DataFrame(data_to_save, columns=[
                "Time", "Knee Angle (deg)", "Hip Angle (deg)", "Back Angle (deg)",
                "Quad Force (N)", "Glute Force (N)", "Hamstring Force (N)", "Back Force (N)",
                "Strain", "Quad Young Modulus (Pa)", "Glute Young Modulus (Pa)",
                "Hamstring Young Modulus (Pa)", "Back Young Modulus (Pa)"
            ])
            csv_data = df_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name="squat_biomechanics_log.csv",
                mime="text/csv",
                help="Download the collected biomechanics data as a CSV file."
            )
            if st.button("Generate & Save Graph", key="graph_btn"):
                # Ensure analysis is not running before generating graph
                if not st.session_state.analysis_running: 
                    generate_and_display_graph(df_log, st.session_state.constants['age_years'])
                else:
                    st.warning("Please stop the live analysis before generating a graph.")
        else:
            st.write("No data collected yet to save.")

    # --- Main Content Area ---
    st.subheader("Live Camera Feed / Processed Video & Real-time Data")
    
    # Use a placeholder for real-time metrics
    metrics_placeholder = st.empty()
    video_placeholder = st.empty() # Placeholder for video display

    if analysis_mode == "Live Camera Analysis":
        ctx = webrtc_streamer(
            key="squat-analysis",
            video_processor_factory=lambda: SquatBiomechanicsProcessor(current_constants),
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    {"urls": ["stun:global.stun.twilio.com:3478"]},
                    {"urls": ["stun:stun.nextcloud.com:3478"]},
                    {"urls": ["stun:stun.voip.blackberry.com:3478"]},
                    {"urls": ["stun:stun.sipgate.net:10000"]},
                    {"urls": ["stun:stun.ekiga.net:3478"]},
                    {"urls": ["stun:stun.ideasip.com:3478"]},
                    {"urls": ["stun:stun.schlund.de:3478"]},
                    {"urls": ["stun:stun.rixtelecom.se:3478"]},
                    {"urls": ["stun:stun.iptel.org:3478"]},
                ]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.state.playing:
            with metrics_placeholder.container():
                st.markdown("---")
                st.subheader("📊 Real-time Biomechanics Data (Live)")
                display_data = st.session_state.get('data_to_display', {
                    "Knee Angle": 0, "Hip Angle": 0, "Back Angle": 0,
                    "Quad F": 0, "Glute F": 0, "Hamstring F": 0, "Back F": 0,
                    "Strain": 0,
                    "Quad E": 0, "Glute E": 0, "Hamstring E": 0, "Back E": 0
                })
                
                col_angle1, col_angle2, col_angle3 = st.columns(3)
                with col_angle1:
                    st.metric("Knee Angle (deg)", f"{display_data.get('Knee Angle', 'N/A')}")
                with col_angle2:
                    st.metric("Hip Angle (deg)", f"{display_data.get('Hip Angle', 'N/A')}")
                with col_angle3:
                    st.metric("Back Angle (deg)", f"{display_data.get('Back Angle', 'N/A')}")

                st.markdown("---")
                col_force1, col_force2, col_force3, col_force4 = st.columns(4)
                with col_force1:
                    st.metric("Quad Force (N)", f"{display_data.get('Quad F', 'N/A')}")
                with col_force2:
                    st.metric("Glute Force (N)", f"{display_data.get('Glute F', 'N/A')}")
                with col_force3:
                    st.metric("Hamstring Force (N)", f"{display_data.get('Hamstring F', 'N/A')}")
                with col_force4:
                    st.metric("Back Force (N)", f"{display_data.get('Back F', 'N/A')}")
                
                st.markdown("---")
                st.metric("Strain (%)", f"{display_data.get('Strain', 'N/A'):.2f}")

                st.markdown("---")
                col_modulus1, col_modulus2, col_modulus3, col_modulus4 = st.columns(4)
                with col_modulus1:
                    st.metric("Quad Young Modulus (MPa)", f"{display_data.get('Quad E', 'N/A'):.2f}")
                with col_modulus2:
                    st.metric("Glute Young Modulus (MPa)", f"{display_data.get('Glute E', 'N/A'):.2f}")
                with col_modulus3:
                    st.metric("Hamstring Young Modulus (MPa)", f"{display_data.get('Hamstring E', 'N/A'):.2f}")
                with col_modulus4:
                    st.metric("Back Young Modulus (MPa)", f"{display_data.get('Back E', 'N/A'):.2f}")
                st.markdown("---")
        else:
            st.info("Click 'Start Live Analysis' in the sidebar and allow webcam access to begin.")
            st.session_state.analysis_running = False

            st.markdown("---")
            st.subheader("Troubleshooting Camera Connection")
            st.write("""
                If your camera feed is not appearing or you see a "Connection is taking longer than expected" message, please try the following:
                1.  **Check your internet connection:** Ensure it's stable and strong.
                2.  **Allow camera access:** Make sure your browser has permission to access your webcam. Look for a pop-up or icon in your browser's address bar.
                3.  **Close other apps using the camera:** Only one application can use the camera at a time.
                4.  **Try a different browser:** Sometimes, browser-specific settings can interfere.
                5.  **Test from a different network:** If possible, try accessing the app from a different Wi-Fi network or a mobile hotspot. Strict network firewalls can block real-time video connections.
                6.  **Restart the app:** Refresh the Streamlit page in your browser.
            """)
    else: # analysis_mode == "Upload Video for Analysis"
        if st.session_state.processed_video_frames:
            st.subheader("Processed Video Playback")
            # Display processed video frames
            # For simplicity, we'll display the first frame and then the data.
            # A full video player would require iterating through frames or a custom component.
            st.image(st.session_state.processed_video_frames[0], channels="RGB", use_column_width=True)
            st.write("*(Note: For full video playback, consider downloading the processed video or using a dedicated video player.)*")
            
            st.subheader("📊 Biomechanics Data from Uploaded Video")
            df_processed_log = pd.DataFrame(st.session_state.processed_video_log_data, columns=[
                "Time", "Knee Angle (deg)", "Hip Angle (deg)", "Back Angle (deg)",
                "Quad Force (N)", "Glute Force (N)", "Hamstring Force (N)", "Back Force (N)",
                "Strain", "Quad Young Modulus (Pa)", "Glute Young Modulus (Pa)",
                "Hamstring Young Modulus (Pa)", "Back Young Modulus (Pa)"
            ])
            st.dataframe(df_processed_log) # Display as a dataframe

            # Display real-time like metrics for the last frame of the processed video
            if not df_processed_log.empty:
                display_data = df_processed_log.iloc[-1].to_dict() # Get last row as dict for display
                st.markdown("---")
                st.subheader("Last Frame Biomechanics Summary")
                col_angle1, col_angle2, col_angle3 = st.columns(3)
                with col_angle1:
                    st.metric("Knee Angle (deg)", f"{display_data.get('Knee Angle (deg)', 'N/A'):.0f}")
                with col_angle2:
                    st.metric("Hip Angle (deg)", f"{display_data.get('Hip Angle (deg)', 'N/A'):.0f}")
                with col_angle3:
                    st.metric("Back Angle (deg)", f"{display_data.get('Back Angle (deg)', 'N/A'):.0f}")

                st.markdown("---")
                col_force1, col_force2, col_force3, col_force4 = st.columns(4)
                with col_force1:
                    st.metric("Quad Force (N)", f"{display_data.get('Quad Force (N)', 'N/A'):.0f}")
                with col_force2:
                    st.metric("Glute Force (N)", f"{display_data.get('Glute Force (N)', 'N/A'):.0f}")
                with col_force3:
                    st.metric("Hamstring Force (N)", f"{display_data.get('Hamstring Force (N)', 'N/A'):.0f}")
                with col_force4:
                    st.metric("Back Force (N)", f"{display_data.get('Back Force (N)', 'N/A'):.0f}")
                
                st.markdown("---")
                st.metric("Strain (%)", f"{display_data.get('Strain', 'N/A'):.2f}")

                st.markdown("---")
                col_modulus1, col_modulus2, col_modulus3, col_modulus4 = st.columns(4)
                with col_modulus1:
                    st.metric("Quad Young Modulus (MPa)", f"{display_data.get('Quad Young Modulus (Pa)', 'N/A') / 1e6:.2f}")
                with col_modulus2:
                    st.metric("Glute Young Modulus (MPa)", f"{display_data.get('Glute Young Modulus (Pa)', 'N/A') / 1e6:.2f}")
                with col_modulus3:
                    st.metric("Hamstring Young Modulus (MPa)", f"{display_data.get('Hamstring Young Modulus (Pa)', 'N/A') / 1e6:.2f}")
                with col_modulus4:
                    st.metric("Back Young Modulus (MPa)", f"{display_data.get('Back Young Modulus (Pa)', 'N/A') / 1e6:.2f}")
                st.markdown("---")

        else:
            st.info("Upload a video file in the sidebar to analyze your squat.")


def generate_and_display_graph(df_log: pd.DataFrame, age: int):
    """Generates and displays biomechanics graphs."""
    if df_log.shape[0] < 2:
        st.warning("Not enough data points collected to create a meaningful graph.")
        return

    timestamps_raw = df_log.iloc[:, 0].values
    time_elapsed = timestamps_raw - timestamps_raw[0]

    # Apply smoothing to Force and Young's Modulus data
    # Define columns to smooth (indices 4-7 for Forces, 9-12 for Young's Moduli)
    # Using a rolling mean with a window of 5 frames
    smoothing_window = 5 # You can adjust this value (e.g., 3, 7, 9)
    for col_idx in range(4, 8): # Forces
        df_log.iloc[:, col_idx] = df_log.iloc[:, col_idx].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    for col_idx in range(9, 13): # Young's Moduli
        df_log.iloc[:, col_idx] = df_log.iloc[:, col_idx].rolling(window=smoothing_window, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16), sharex=True)
    fig.suptitle('Squat Biomechanics Analysis Over Time', fontsize=16)

    # Joint Angles
    ax0 = axes[0]
    ax0.plot(time_elapsed, df_log.iloc[:, 1], label="Knee Angle")
    ax0.plot(time_elapsed, df_log.iloc[:, 2], label="Hip Angle")
    ax0.plot(time_elapsed, df_log.iloc[:, 3], label="Back Angle")
    ax0.set_ylabel('Angle (deg)')
    ax0.legend()
    ax0.grid(True)
    ax0.set_title('Joint Angles')

    # Muscle Forces
    ax1 = axes[1]
    ax1.plot(time_elapsed, df_log.iloc[:, 4], label="Quad Force")
    ax1.plot(time_elapsed, df_log.iloc[:, 5], label="Glute Force")
    ax1.plot(time_elapsed, df_log.iloc[:, 6], label="Hamstring Force")
    ax1.plot(time_elapsed, df_log.iloc[:, 7], label="Back Force")
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Muscle Forces')

    # Strain
    ax2 = axes[2]
    ax2.plot(time_elapsed, df_log.iloc[:, 8] * 100, label="Calculated Strain (%)", color='purple') # Changed label
    ax2.set_ylabel('Strain (%)')
    ax2.grid(True)
    ax2.set_title('Strain')
    
    # Add horizontal line for estimated physiological strain limit
    max_strain_limit = _estimate_max_physiological_strain(age) # Use 'age' parameter directly
    ax2.axhline(y=max_strain_limit, color='red', linestyle='--', label=f'Estimated Max Strain for Age {age}: {max_strain_limit:.1f}%') # Use 'age' parameter directly
    ax2.legend()

    # Young's Moduli
    ax3 = axes[3]
    ax3.plot(time_elapsed, df_log.iloc[:, 9] / 1e6, label="Quad Young Modulus")
    ax3.plot(time_elapsed, df_log.iloc[:, 10] / 1e6, label="Glute Young Modulus")
    ax3.plot(time_elapsed, df_log.iloc[:, 11] / 1e6, label="Hamstring Young Modulus")
    ax3.plot(time_elapsed, df_log.iloc[:, 12] / 1e6, label="Back Young Modulus")
    ax3.set_ylabel('Young\'s Modulus (MPa)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Young\'s Moduli')
    ax3.set_xlabel('Time (seconds)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    st.pyplot(fig) # Display the plot in Streamlit

    # Save the figure to a BytesIO object for download
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download Graph as PNG",
        data=buf.getvalue(),
        file_name="squat_biomechanics_graph.png",
        mime="image/png",
        help="Download the generated graph as a PNG image."
    )
    plt.close(fig) # Ensure the figure is closed after saving and displaying

if __name__ == "__main__":
    main()
