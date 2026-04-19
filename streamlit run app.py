import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 🎨 STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Premium Dark Mode Look
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #334155; margin-bottom: 15px; }
    .metric-value { font-size: 48px; font-weight: bold; color: #3b82f6; }
    .goal-value { font-size: 48px; font-weight: bold; color: #10b981; }
    .status-good { color: #10b981; font-size: 24px; font-weight: bold; }
    .status-warn { color: #ef4444; font-size: 24px; font-weight: bold; }
    .status-idle { color: #f59e0b; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧩 YOUR CUSTOM AI FUNCTIONS
# ==========================================
def get_branch_features(x, landmark_indices):
    feature_indices = []
    for i in landmark_indices: feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
    return tf.gather(x, feature_indices, axis=2)

def get_angle_features(x): return tf.gather(x, [132, 133], axis=2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================
# 🚀 LOAD AI MODEL
# ==========================================
@st.cache_resource
def load_ai():
    try:
        model = load_model("squat_specialist_v3.h5", custom_objects={'get_branch_features': get_branch_features, 'get_angle_features': get_angle_features, 'tf': tf})
        return model
    except Exception:
        return None

model = load_ai()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 🌐 STREAMLIT UI LAYOUT
# ==========================================
st.title("🟦 POSEGURU | Clinical Rehab AI")

# Session State for Target Angle
if 'target_angle' not in st.session_state:
    st.session_state.target_angle = 120.0

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Tracking")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Clinical Metrics")
    reps_placeholder = st.empty()
    angle_placeholder = st.empty()
    
    goal_placeholder = st.empty()
    goal_placeholder.markdown(f"""
        <div class="metric-card">
            <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
            <p class="goal-value">{int(st.session_state.target_angle)}°</p>
        </div>
    """, unsafe_allow_html=True)
    
    start_btn = st.button("▶ START CAMERA", use_container_width=True)
    stop_btn = st.button("⏹ STOP CAMERA", use_container_width=True)
    cal_btn = st.button("🎯 CALIBRATE SAFE DEPTH", use_container_width=True)

# Handle Calibration outside the loop so it updates immediately
if cal_btn and 'current_knee_angle' in st.session_state:
    st.session_state.target_angle = st.session_state.current_knee_angle
    goal_placeholder.markdown(f"""
        <div class="metric-card">
            <p style="color: #94a3b8; letter-spacing: 2px;">ADAPTED GOAL</p>
            <p class="goal-value">{int(st.session_state.target_angle)}°</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🎥 THE AI CAMERA LOOP
# ==========================================
if start_btn:
    cap = cv2.VideoCapture(0)
    
    sequence_buffer = []
    squat_state = "up"
    reps = 0
    frame_count = 0
    st.session_state.current_knee_angle = 180.0

    while cap.isOpened() and not stop_btn:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        status_text = "WAITING..."
        status_class = "status-idle"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
                status_text = "BACK UP (SHOW LEGS)"
                status_class = "status-warn"
                sequence_buffer.clear()
            else:
                features = []
                for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
                current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
                st.session_state.current_knee_angle = current_knee_angle # Save to session state for calibration
                
                target = st.session_state.target_angle

                features.extend([current_knee_angle, current_hip_angle])
                features.extend([0.0] * (240 - len(features)))
                sequence_buffer.append(features)
                sequence_buffer = sequence_buffer[-30:]

                if len(sequence_buffer) == 30 and frame_count % 4 == 0 and model is not None:
                    class_id = np.argmax(model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0))

                if current_knee_angle > 170 and current_hip_angle > 165:
                    status_text = "IDLE"
                    status_class = "status-idle"
                    if squat_state == "down": 
                        reps += 1
                        squat_state = "up"
                else:
                    squat_state = "down"
                    if current_hip_angle < 100: 
                        status_text = "STRAIGHTEN YOUR BACK"
                        status_class = "status-warn"
                    elif current_knee_angle > (target + 5):
                        status_text = "GO DOWN SLIGHTLY"
                        status_class = "status-warn"
                    else:
                        status_text = "GOOD (ADAPTED REHAB)"
                        status_class = "status-good"

        # Update Video UI Live
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Text UI Live
        status_placeholder.markdown(f'<div style="text-align:center; padding: 10px; background-color: #1e293b; border-radius: 10px;"><span class="{status_class}">STATUS: {status_text}</span></div>', unsafe_allow_html=True)
        reps_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">REPETITIONS</p><p class="metric-value">{reps}</p></div>', unsafe_allow_html=True)
        angle_placeholder.markdown(f'<div class="metric-card"><p style="color: #94a3b8; letter-spacing: 2px;">LIVE KNEE ANGLE</p><p class="metric-value">{int(st.session_state.current_knee_angle)}°</p></div>', unsafe_allow_html=True)

    cap.release()