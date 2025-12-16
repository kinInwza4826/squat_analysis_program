import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ======================================================
# CONSTANTS & BIOMECHANICAL FUNCTIONS
# ======================================================

# Biomechanical constants (DO NOT CHANGE)
wH_ratio = 0.07
wA_ratio = 0.12
wT_ratio = 0.45
dH = 0.72
dA = 0.48
dT = 0.36
dM = 0.48
muscle_offset = 12  # degrees
SEG_ANGLE = 60      # fixed constant
L0 = 0.01
A = 0.0015
E = 5e6

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def trunk_angle(shoulder, hip):
    """
    Calculates the trunk angle from the vertical.
    Returns: angle in degrees (upright=90°, horizontal=0°).
    """
    sx, sy = shoulder
    hx, hy = hip

    vx = sx - hx
    vy = sy - hy

    trunk = np.array([vx, vy])
    vertical = np.array([0, -1])

    dot = np.dot(trunk, vertical)
    mag_trunk = np.linalg.norm(trunk)
    mag_vertical = np.linalg.norm(vertical) + 1e-6 # Avoid div by zero

    # Calculate angle from the vertical
    angle_rad = np.arccos(np.clip(dot / (mag_trunk * mag_vertical), -1, 1))
    angle_from_vertical = np.degrees(angle_rad)

    # Convert to the desired convention: 90° = upright, 0° = horizontal
    return 90 - angle_from_vertical


def compute_biomechanics(angle_deg, w):
    """
    Calculates Muscle Force (Fm), Vertebral Force (Fv), and Disc Deformation.
    'w' is the total body weight in Newtons.
    """
    wH = wH_ratio * w
    wA = wA_ratio * w
    wT = wT_ratio * w

    # Calculate muscle force (Fm) - M_load = M_muscle
    numerator = (
        dH * math.sin(math.radians(SEG_ANGLE)) * wH +
        dA * math.sin(math.radians(SEG_ANGLE)) * wA +
        dT * math.sin(math.radians(SEG_ANGLE)) * wT
    )
    denominator = dM * math.sin(math.radians(muscle_offset))

    # Handle division by zero if muscle_offset is 0, though unlikely with 12 deg
    if denominator == 0:
        Fm = 0
    else:
        Fm = numerator / denominator

    # Calculate Force Components (Fx, Fy)
    muscle_angle = math.radians(angle_deg - muscle_offset)

    Fx = Fm * math.cos(muscle_angle)
    # Sum of vertical forces: Muscle vertical component + segment weights
    Fy = Fm * math.sin(muscle_angle) + (wH + wA + wT) 

    # Vertebral Compression Force (Fv)
    Fv = math.sqrt(Fx**2 + Fy**2)

    # Disc Deformation (based on Hooke's Law for elastic material)
    deformation = (Fv * L0) / (A * E)

    return Fm, Fv, deformation


# ======================================================
# STREAMLIT APP LAYOUT
# ======================================================

st.set_page_config(layout="wide", page_title="Spinal Biomechanics Analysis")

st.title("🏋️ Real-Time Spinal Biomechanics Analysis")
st.markdown("Analyze the forces and deformation in the lumbar spine during bending using a simplified biomechanical model and MediaPipe for pose estimation.")
# Removed the placeholder image tag that caused a SyntaxError: 
---

# 1. User Input and Controls
col_in, col_ctrl = st.columns([1, 1])

with col_in:
    # Get mass input from user
    mass_kg = st.number_input(
        "Enter your body mass (kg):",
        min_value=1.0,
        max_value=300.0,
        value=70.0,
        step=1.0,
        format="%.1f"
    )
    w = mass_kg * 9.81  # Calculate body weight force (Newtons)
    st.info(f"Calculated Body Weight (w): **{w:.2f} N**")

with col_ctrl:
    # Camera Control Buttons
    st.subheader("Camera Control")
    # Using session state to manage button clicks (run/stop)
    if 'running' not in st.session_state:
        st.session_state.running = False
        st.session_state.history = {'timestamps': [], 'angles': [], 'forces': [], 'deforms': []}

    col_start, col_stop = st.columns(2)
    
    # Start button logic
    if col_start.button("▶️ Start Camera", key="start_btn", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.history = {'timestamps': [], 'angles': [], 'forces': [], 'deforms': []} # Reset history on start
    
    # Stop button logic
    if col_stop.button("⏹️ Stop/Reset Analysis", key="stop_btn", disabled=not st.session_state.running):
        st.session_state.running = False
        st.session_state.history = {'timestamps': [], 'angles': [], 'forces': [], 'deforms': []}


# 2. Real-Time Video and Metrics Display
st.header("📊 Live Analysis & Metrics")
col_vid, col_metrics = st.columns([2, 1])

# Placeholders for video, metrics, and charts
video_placeholder = col_vid.empty()
metrics_placeholder = col_metrics.container()
chart_placeholder = st.empty()


# 3. Live Processing Loop (Only runs if 'running' state is True)
if st.session_state.running:
    
    # Initialize/reset history if not running, this is redundant but ensures clean state
    if not st.session_state.history['timestamps']:
        st.session_state.history = {'timestamps': [], 'angles': [], 'forces': [], 'deforms': []}

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from camera. Is the camera in use by another app?")
                st.session_state.running = False
                break

            h, w_img, _ = frame.shape
            # Flip the frame for a more intuitive mirror-like view
            frame = cv2.flip(frame, 1) 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False # Read-only for performance
            results = pose.process(rgb)
            rgb.flags.writeable = True # Writeable again
            
            # Initialize values for display
            camera_angle, Fv, deformation = 0.0, 0.0, 0.0
            tracking_status = "Detecting..."
            color = (0, 0, 255) # Red for initial/lost

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                try:
                    # Use landmarks from the left side (or average both)
                    shoulder = (
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w_img,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                    )
                    hip = (
                        lm[mp_pose.PoseLandmark.LEFT_HIP].x * w_img,
                        lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                    )
    
                    # 1. Angle Calculation
                    camera_angle = trunk_angle(shoulder, hip)
                    bio_angle = 90 - camera_angle # 0°=upright, 30°=bent
    
                    # 2. Biomechanical Calculation
                    Fm, Fv, deformation = compute_biomechanics(bio_angle, w)
    
                    # 3. Data Logging
                    t = time.time() - start_time
                    st.session_state.history['angles'].append(camera_angle)
                    st.session_state.history['forces'].append(Fv)
                    st.session_state.history['deforms'].append(deformation)
                    st.session_state.history['timestamps'].append(t)
                    tracking_status = "Tracking OK"
                    color = (0, 255, 0) # Green for OK

                    # 4. Draw Skeleton
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
                    )
                except Exception:
                    tracking_status = "Lost: Stand upright and face camera"
                    color = (0, 165, 255) # Orange for lost
                    
            # Add tracking status to the frame
            cv2.putText(frame, tracking_status, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
            # 5. Display Frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # 6. Display Metrics
            with metrics_placeholder:
                # Clear previous metrics and display new ones
                metrics_placeholder.empty()
                st.metric("Trunk Angle", f"{camera_angle:.1f} °", help="90° is fully upright, 0° is horizontal")
                st.metric("Vertebral Force ($F_v$)", f"{Fv:.1f} N", help="The compressive force on the lumbar disc.")
                st.metric("Disc Deformation", f"{deformation*1000:.3f} mm", help="The estimated change in disc height.")

            # 7. Update Charts (only update every ~5 frames for performance)
            if len(st.session_state.history['timestamps']) % 5 == 0 and len(st.session_state.history['timestamps']) > 1:
                with chart_placeholder:
                    chart_placeholder.empty()
                    
                    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
                    
                    # Angle Chart
                    axes[0].plot(st.session_state.history['timestamps'], st.session_state.history['angles'], color='green')
                    axes[0].set_title("Bending Angle Over Time")
                    axes[0].set_ylabel("Angle (deg)")
                    axes[0].set_ylim(0, 100)
                    
                    # Force Chart
                    axes[1].plot(st.session_state.history['timestamps'], st.session_state.history['forces'], color='red')
                    axes[1].set_title("Vertebral Force $F_v$ (Newtons)")
                    axes[1].set_ylabel("Force (N)")
                    
                    # Deformation Chart
                    axes[2].plot(st.session_state.history['timestamps'], np.array(st.session_state.history['deforms']) * 1000, color='blue')
                    axes[2].set_title("Disc Deformation Over Time")
                    axes[2].set_xlabel("Time (s)")
                    axes[2].set_ylabel("Deformation (mm)")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        # Cleanup after loop ends (stop button pressed or camera error)
        cap.release()
        st.session_state.running = False
        st.warning("Camera feed stopped.")

elif not st.session_state.running:
    # Initial state or after Stop is pressed
    video_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Press 'Start Camera' to begin analysis.", use_column_width=True)
    metrics_placeholder.empty()
    chart_placeholder.empty()
