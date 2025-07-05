import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from aiortc.mediastreams import VideoFrame # Corrected import for VideoFrame
from typing import Union, List, Dict, Any

# --- Biomechanics Calculation Functions (from original app) ---
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
    return tau / (r * math.cos(theta))

def _calculate_young_modulus(F, L0, A, delta_L):
    """Calculates Young's Modulus based on force, original length, area, and change in length."""
    # Prevent division by zero
    if A * delta_L == 0:
        return 0
    return (F * L0) / (A * delta_L)

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

# --- Video Processor Class for Streamlit-WebRTC ---
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
                # Use st.warning for messages that appear in the Streamlit app
                st.warning(f"Partial landmark detection or error: {e}")
                # Update display data even on error
                if 'data_to_display' in st.session_state: # Check before updating
                    st.session_state.data_to_display = data_to_display
                # Draw landmarks if available, even if calculation failed
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

            F_quad = _calculate_force(tau_knee, self.constants['r_quad'], theta_knee)
            F_glute = _calculate_force(tau_hip, self.constants['r_glute'], theta_hip)
            F_hamstring = _calculate_force(tau_hip, self.constants['r_hamstring_hip'], theta_hip)
            F_erector_spinae = _calculate_force(tau_hip, self.constants['r_erector_spinae'], theta_hip)

            delta_L = self.constants['L0'] * self.constants['delta_L_ratio']
            strain = delta_L / self.constants['L0'] if self.constants['L0'] != 0 else 0

            E_quad = _calculate_young_modulus(F_quad, self.constants['L0'], self.constants['A_quad'], delta_L)
            E_glute = _calculate_young_modulus(F_glute, self.constants['L0'], self.constants['A_glute'], delta_L)
            E_hamstring = _calculate_young_modulus(F_hamstring, self.constants['L0'], self.constants['A_hamstring'], delta_L)
            E_erector_spinae = _calculate_young_modulus(F_erector_spinae, self.constants['L0'], self.constants['A_erector_spinae'], delta_L)

            data_to_display = {
                "Knee Angle": int(knee_angle),
                "Hip Angle": int(hip_angle),
                "Back Angle": int(back_angle),
                "Quad F": int(F_quad),
                "Glute F": int(F_glute),
                "Hamstring F": int(F_hamstring),
                "Back F": int(F_erector_spinae),
                "Strain": strain * 100, # Convert to percentage
                "Quad E": E_quad / 1e6, # Convert to MPa
                "Glute E": E_glute / 1e6, # Convert to MPa
                "Hamstring E": E_hamstring / 1e6, # Convert to MPa
                "Back E": E_erector_spinae / 1e6 # Convert to MPa
            }

            # Update session state for display in Streamlit UI
            if 'data_to_display' in st.session_state: # Check before updating
                st.session_state.data_to_display = data_to_display

            # Log data if analysis is running
            if st.session_state.get('analysis_running', False) and 'log_data' in st.session_state:
                timestamp = time.time()
                st.session_state.log_data.append([
                    timestamp, knee_angle, hip_angle, back_angle,
                    F_quad, F_glute, F_hamstring, F_erector_spinae,
                    strain, E_quad, E_glute, E_hamstring, E_erector_spinae
                ])

            # Draw landmarks on the image
            self._draw_landmarks(image_bgr, results)

        return VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def _draw_landmarks(self, image, results):
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                      self.mp_pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                      mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                     )

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Real-Time Squat Biomechanics Analysis")
    st.title("🏋️ Real-Time Squat Biomechanics Analysis")

    # Initialize ALL session state variables unconditionally at the very top of main()
    # This ensures they are always present when accessed, especially by the video processor factory.
    if 'constants' not in st.session_state:
        st.session_state.constants = {
            'A_quad': 0.00006,
            'A_glute': 0.00009,
            'A_hamstring': 0.00007,
            'A_erector_spinae': 0.00003,
            'L0': 0.10,
            'r_quad': 0.04,
            'r_glute': 0.07,
            'r_hamstring_hip': 0.06,
            'r_erector_spinae': 0.05,
            'r_external': 0.35,
            'weight_kg': 100.0,
            'phi_deg': 15.0,
            'phi': math.radians(15.0), # Calculated from phi_deg
            'delta_L_ratio': 0.08,
            'age_years': 30
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

    # Create a local copy of constants to pass to the video processor factory.
    # This ensures the dictionary is fully initialized when the lambda is defined.
    # This copy is made *after* st.session_state.constants is guaranteed to be set.
    current_constants = st.session_state.constants.copy()

    # --- Sidebar for Controls and Constants ---
    with st.sidebar:
        st.header("⚙️ Settings & Controls")

        st.subheader("Basic Constants")
        # Update session state constants directly via widgets
        st.session_state.constants['age_years'] = st.number_input(
            "Age (years):",
            min_value=0, max_value=150, value=st.session_state.constants['age_years'], step=1,
            help="Your age in years. Used to estimate a general physiological strain limit for tendons."
        )
        st.session_state.constants['weight_kg'] = st.number_input(
            "Weight (kg):",
            min_value=0.0, value=st.session_state.constants['weight_kg'], step=1.0, format="%.1f",
            help="Enter the total weight being lifted (body weight + barbell weight)."
        )
        st.session_state.constants['r_external'] = st.number_input(
            "External Moment Arm (m):",
            min_value=0.0, value=st.session_state.constants['r_external'], step=0.01, format="%.2f",
            help="Distance from the joint (hip/knee) to the line of action of the external force. Adjust based on barbell position."
        )
        st.session_state.constants['phi_deg'] = st.number_input(
            "Hip Abduction Angle (deg):",
            min_value=0.0, max_value=90.0, value=st.session_state.constants['phi_deg'], step=1.0, format="%.1f",
            help="Angle representing the load's line of action relative to the body's sagital plane. Helps account for non-vertical force components."
        )
        st.session_state.constants['phi'] = math.radians(st.session_state.constants['phi_deg']) # Update phi_rad

        st.session_state.constants['r_quad'] = st.number_input(
            "Quad Moment Arm (m):",
            min_value=0.0, value=st.session_state.constants['r_quad'], step=0.01, format="%.2f",
            help="Effective moment arm of the quadriceps muscle around the knee joint. A key factor in quadriceps force calculation."
        )
        st.session_state.constants['r_glute'] = st.number_input(
            "Glute Moment Arm (m):",
            min_value=0.0, value=st.session_state.constants['r_glute'], step=0.01, format="%.2f",
            help="Effective moment arm of the gluteal muscles around the hip joint. Important for hip extension force."
        )
        st.session_state.constants['r_hamstring_hip'] = st.number_input(
            "Hamstring Moment Arm (m):",
            min_value=0.0, value=st.session_state.constants['r_hamstring_hip'], step=0.01, format="%.2f",
            help="Effective moment arm of the hamstrings around the hip joint (primarily for hip extension in squat)."
        )
        st.session_state.constants['r_erector_spinae'] = st.number_input(
            "Erector Spinae Moment Arm (m):",
            min_value=0.0, value=st.session_state.constants['r_erector_spinae'], step=0.01, format="%.2f",
            help="Effective moment arm of the erector spinae (back muscles) around the hip joint, crucial for maintaining torso uprightness."
        )

        with st.expander("Advanced Anatomical Constants"):
            st.session_state.constants['A_quad'] = st.number_input(
                "Quad Area (m²):",
                min_value=0.0, value=st.session_state.constants['A_quad'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the quadriceps muscle. Larger area generally means higher force potential."
            )
            st.session_state.constants['A_glute'] = st.number_input(
                "Glute Area (m²):",
                min_value=0.0, value=st.session_state.constants['A_glute'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the gluteal muscles. Affects glute force and Young's Modulus."
            )
            st.session_state.constants['A_hamstring'] = st.number_input(
                "Hamstring Area (m²):",
                min_value=0.0, value=st.session_state.constants['A_hamstring'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the hamstring muscles. Influences hamstring force contribution."
            )
            st.session_state.constants['A_erector_spinae'] = st.number_input(
                "Erector Spinae Area (m²):",
                min_value=0.0, value=st.session_state.constants['A_erector_spinae'], step=0.00001, format="%.5f",
                help="Physiological cross-sectional area of the erector spinae muscles. Critical for back extensor force."
            )
            st.session_state.constants['L0'] = st.number_input(
                "Resting Tendon Length (m):",
                min_value=0.0, value=st.session_state.constants['L0'], step=0.01, format="%.2f",
                help="Initial, unstretched length of the muscle-tendon unit. Baseline for strain calculation."
            )
            st.session_state.constants['delta_L_ratio'] = st.number_input(
                "Delta L Ratio:",
                min_value=0.0, max_value=1.0, value=st.session_state.constants['delta_L_ratio'], step=0.01, format="%.2f",
                help="Ratio of the change in tendon length to the resting length. Directly influences the calculated strain."
            )

        st.subheader("Control Analysis")
        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button("Start Analysis", key="start_btn"):
                st.session_state.analysis_running = True
                st.session_state.log_data = [] # Clear previous data on start
                st.success("Analysis started! Perform your squats.")
        with col2_btn:
            if st.button("Stop Analysis", key="stop_btn"):
                st.session_state.analysis_running = False
                st.info("Analysis stopped. You can now save data.")

        st.subheader("Save Options")
        if st.session_state.log_data:
            df_log = pd.DataFrame(st.session_state.log_data, columns=[
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
                if not st.session_state.analysis_running:
                    generate_and_display_graph(df_log, st.session_state.constants['age_years'])
                else:
                    st.warning("Please stop the analysis before generating a graph.")
        else:
            st.write("No data collected yet to save.")

    # --- Main Content Area ---
    st.subheader("Live Camera Feed & Real-time Data")
    
    # Use a placeholder for real-time metrics
    metrics_placeholder = st.empty()

    # Streamlit-WebRTC component for live video
    ctx = webrtc_streamer(
        key="squat-analysis",
        # Pass the local copy of constants to the video processor factory
        # This ensures the dictionary is fully initialized when the lambda is defined.
        video_processor_factory=lambda: SquatBiomechanicsProcessor(current_constants),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Process frames asynchronously
    )

    # Update real-time metrics in the placeholder
    if ctx.state.playing:
        with metrics_placeholder.container():
            st.markdown("---")
            st.subheader("📊 Real-time Biomechanics Data")
            # Ensure display_data is initialized even if not yet populated by processor
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
        st.info("Click 'Start Analysis' in the sidebar and allow webcam access to begin.")
        st.session_state.analysis_running = False # Ensure analysis is off if video is not playing

def generate_and_display_graph(df_log: pd.DataFrame, age: int):
    """Generates and displays biomechanics graphs."""
    if df_log.shape[0] < 2:
        st.warning("Not enough data points collected to create a meaningful graph.")
        return

    timestamps_raw = df_log.iloc[:, 0].values
    time_elapsed = timestamps_raw - timestamps_raw[0]

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
    ax2.plot(time_elapsed, df_log.iloc[:, 8] * 100, label="Strain (%)", color='purple')
    ax2.set_ylabel('Strain (%)')
    ax2.grid(True)
    ax2.set_title('Strain')
    
    # Add horizontal line for estimated physiological strain limit
    max_strain_limit = _estimate_max_physiological_strain(age)
    ax2.axhline(y=max_strain_limit, color='red', linestyle='--', label=f'Estimated Max Strain for Age {age}: {max_strain_limit:.1f}%')
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
    plt.close(fig) # Close the figure to free up memory

if __name__ == "__main__":
    main()
