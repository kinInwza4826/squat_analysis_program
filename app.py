import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ======================================================
# STREAMLIT SETUP
# ======================================================
st.set_page_config(page_title="Biomechanical Analysis", layout="wide")
st.title("📊 Real-time Biomechanical Analysis")

# ======================================================
# USER INPUT
# ======================================================
mass_kg = st.number_input("Enter your body mass (kg)", min_value=30.0, max_value=200.0, value=60.0)
w = mass_kg * 9.81
st.write(f"**Body weight force:** {w:.2f} N")

# ======================================================
# CONSTANTS (FROM PAPER)
# ======================================================
wH_ratio = 0.07
wA_ratio = 0.12
wT_ratio = 0.45

dH = 0.72
dA = 0.48
dT = 0.36
dM = 0.48

muscle_offset = 12
SEG_ANGLE = 60

L0 = 0.01
A = 0.0015
E = 5e6

# ======================================================
# FUNCTIONS
# ======================================================
def trunk_angle(shoulder, hip):
    sx, sy = shoulder
    hx, hy = hip
    trunk = np.array([sx - hx, sy - hy])
    vertical = np.array([0, -1])
    dot = np.dot(trunk, vertical)
    mag = np.linalg.norm(trunk) + 1e-6
    angle_from_vertical = np.degrees(np.arccos(np.clip(dot/mag, -1, 1)))
    return 90 - angle_from_vertical

def compute_biomechanics(angle_deg):
    wH = wH_ratio * w
    wA = wA_ratio * w
    wT = wT_ratio * w

    numerator = (
        dH * math.sin(math.radians(SEG_ANGLE)) * wH +
        dA * math.sin(math.radians(SEG_ANGLE)) * wA +
        dT * math.sin(math.radians(SEG_ANGLE)) * wT
    )
    denominator = dM * math.sin(math.radians(muscle_offset))
    Fm = numerator / denominator

    muscle_angle = math.radians(angle_deg - muscle_offset)
    Fx = Fm * math.cos(muscle_angle)
    Fy = Fm * math.sin(muscle_angle) + (wH + wA + wT)
    Fv = math.sqrt(Fx**2 + Fy**2)

    deformation = (Fv * L0) / (A * E)
    return Fm, Fv, deformation

# ======================================================
# SESSION STATE
# ======================================================
if "run" not in st.session_state:
    st.session_state.run = False

if "data" not in st.session_state:
    st.session_state.data = {
        "time": [], "angle": [], "force": [], "deform": []
    }

# ======================================================
# BUTTONS
# ======================================================
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start"):
        st.session_state.run = True
with col2:
    if st.button("⏹ Stop"):
        st.session_state.run = False

# ======================================================
# VIDEO + ANALYSIS
# ======================================================
frame_window = st.image([])
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

if st.session_state.run:

    cap = cv2.VideoCapture(0)
    start = time.time()

    with mp_pose.Pose() as pose:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            h, w_img, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                shoulder = (
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w_img,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                )
                hip = (
                    lm[mp_pose.PoseLandmark.LEFT_HIP].x * w_img,
                    lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                )

                camera_angle = trunk_angle(shoulder, hip)
                bio_angle = 90 - camera_angle

                Fm, Fv, deform = compute_biomechanics(bio_angle)

                t = time.time() - start
                st.session_state.data["time"].append(t)
                st.session_state.data["angle"].append(camera_angle)
                st.session_state.data["force"].append(Fv)
                st.session_state.data["deform"].append(deform)

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(frame, f"Angle: {camera_angle:.1f} deg", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Fv: {Fv:.1f} N", (20,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

# ======================================================
# GRAPHS
# ======================================================
if len(st.session_state.data["time"]) > 10:
    st.subheader("📈 Results")

    fig1, ax1 = plt.subplots()
    ax1.plot(st.session_state.data["time"], st.session_state.data["angle"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (deg)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(st.session_state.data["time"], st.session_state.data["force"])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force (N)")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(
        st.session_state.data["time"],
        np.array(st.session_state.data["deform"]) * 1000
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Deformation (mm)")
    st.pyplot(fig3)
