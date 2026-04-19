import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import subprocess
import sys

from exercise_config import (
    EXERCISES, EXPECTED_FEATURES, SEQUENCE_LENGTH,
    BRANCH_INDICES, ANGLE_FEATURE_INDICES,
    compute_all_angles, build_feature_vector, get_exercise_list
)

# ==========================================
# 📱 SET YOUR CAMERA SOURCE HERE
# ==========================================
CAMERA_SOURCE = 0  # Use 0 for default webcam, or URL for IP camera
# CAMERA_SOURCE = "http://192.0.0.4:8080/video"

CALIBRATION_FILE = "user_calibration.json"

# ==========================================
# 🗣️ THE BULLETPROOF AUDIO COACH
# ==========================================
def speak(text):
    try:
        script = f"""
import pyttsx3
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.say("{text}")
    engine.runAndWait()
except:
    pass
"""
        subprocess.Popen([sys.executable, "-c", script], creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception:
        pass

# ==========================================
# 🧩 CUSTOM AI FUNCTIONS (for model loading)
# ==========================================
def get_branch_features(x, landmark_indices):
    feature_indices = []
    for i in landmark_indices:
        feature_indices.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
    return tf.gather(x, feature_indices, axis=2)

def get_angle_features(x):
    return tf.gather(x, ANGLE_FEATURE_INDICES, axis=2)

# ==========================================
# 🎯 CALIBRATION HELPERS
# ==========================================
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return {}

# ==========================================
# 🎨 STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PoseGuru AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0;
        font-size: 16px;
    }

    /* Premium Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #f8fafc;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff;
        border-color: #60a5fa;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4), 0 4px 15px rgba(59, 130, 246, 0.3);
        transform: translateY(-1px);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    .metric-label {
        color: #94a3b8;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 44px;
        font-weight: 800;
        color: #3b82f6;
        line-height: 1.1;
    }
    .goal-value {
        font-size: 44px;
        font-weight: 800;
        color: #10b981;
        line-height: 1.1;
    }

    /* Status Text */
    .status-good { color: #10b981; font-size: 22px; font-weight: 700; }
    .status-warn { color: #ef4444; font-size: 22px; font-weight: 700; }
    .status-idle { color: #f59e0b; font-size: 22px; font-weight: 700; }

    /* Exercise Card (in sidebar) */
    .exercise-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #475569;
        margin-bottom: 10px;
    }
    .exercise-card h4 { color: #e2e8f0; margin: 0 0 4px 0; }
    .exercise-card p { color: #94a3b8; margin: 0; font-size: 13px; }

    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Angle display card */
    .angle-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 10px 16px;
        border-radius: 10px;
        border-left: 3px solid #3b82f6;
        margin-bottom: 8px;
    }
    .angle-name { color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; }
    .angle-val { color: #e2e8f0; font-size: 28px; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 🏋️ SIDEBAR: EXERCISE SELECTOR
# ==========================================
with st.sidebar:
    st.markdown('<p class="main-title">🏋️ POSEGURU AI</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 Select Exercise")

    exercise_keys = get_exercise_list()
    exercise_labels = [EXERCISES[k]["display_name"] for k in exercise_keys]

    selected_label = st.radio(
        "Choose your exercise:",
        exercise_labels,
        index=0,
        label_visibility="collapsed"
    )

    # Map display name back to key
    selected_key = exercise_keys[exercise_labels.index(selected_label)]
    exercise = EXERCISES[selected_key]

    st.markdown("---")
    st.markdown(f"""
        <div class="exercise-card">
            <h4>{exercise['display_name']}</h4>
            <p>{exercise['description']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Show tracked angles
    st.markdown("#### 📐 Tracked Angles")
    for label, angle_key in exercise["display_angles"].items():
        ranges = exercise["correct_ranges"].get(angle_key, {})
        range_text = f"{ranges.get('min', '?')}°–{ranges.get('max', '?')}°" if ranges else "—"
        st.markdown(f"- **{label}**: {range_text}")

    st.markdown("---")

    # Model status
    model_file = f"{exercise['csv_prefix'].lower()}_specialist.h5"
    if os.path.exists(model_file):
        st.success(f"✅ Model loaded: {model_file}")
    else:
        st.warning(f"⚠️ No model: {model_file}")
        st.caption("Train with: `python train_multi_exercise.py`")

# ==========================================
# 🧠 LOAD AI MODEL FOR SELECTED EXERCISE
# ==========================================
@st.cache_resource
def load_exercise_model(model_path):
    try:
        model = load_model(model_path, custom_objects={
            'get_branch_features': get_branch_features,
            'get_angle_features': get_angle_features,
            'tf': tf
        })
        return model
    except Exception:
        return None

model = None
if os.path.exists(model_file):
    model = load_exercise_model(model_file)

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 🌐 MAIN UI LAYOUT
# ==========================================
st.markdown(f'<span class="main-title">POSEGURU</span> | {exercise["display_name"]}', unsafe_allow_html=True)

# Load calibration
cal_data = load_calibration()
cal_config = exercise["calibration"]

if 'target_angles' not in st.session_state:
    st.session_state.target_angles = {}

# Set target for selected exercise
if selected_key in cal_data:
    st.session_state.target_angles[selected_key] = float(cal_data[selected_key])
elif selected_key not in st.session_state.target_angles:
    st.session_state.target_angles[selected_key] = float(cal_config["default_target"])

current_target = st.session_state.target_angles.get(selected_key, cal_config["default_target"])

# Camera state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Tracking")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Clinical Metrics")
    reps_placeholder = st.empty()

    # Dynamic angle placeholders based on selected exercise
    angle_placeholders = {}
    for label in exercise["display_angles"]:
        angle_placeholders[label] = st.empty()

    goal_placeholder = st.empty()
    goal_placeholder.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">🎯 {exercise['name']} Goal</p>
            <p class="goal-value">{int(current_target)}°</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # Spacer
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_btn = st.button("▶ START", use_container_width=True)
    with btn_col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)

    cal_btn = st.button(f"🎯 CALIBRATE {exercise['name'].upper()}", use_container_width=True)

    if start_btn:
        st.session_state.camera_active = True
        speak(f"{exercise['name']} mode started. Ready when you are.")

    if stop_btn:
        st.session_state.camera_active = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        speak("Workout paused.")

# ==========================================
# 🎯 CALIBRATION TRIGGER
# ==========================================
if cal_btn:
    st.session_state.camera_active = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    with st.spinner(f"Calibrating {exercise['name']}... Follow the OpenCV window!"):
        speak(f"Calibrating {exercise['name']}. Get ready.")
        time.sleep(2)

        # Launch calibration for selected exercise
        subprocess.run([sys.executable, "calibrate_user.py", "--exercise", selected_key])

        # Reload calibration
        new_cal = load_calibration()
        if selected_key in new_cal:
            new_target = float(new_cal[selected_key])
            st.session_state.target_angles[selected_key] = new_target
            speak(f"Calibration complete. Target is {int(new_target)} degrees.")

    st.rerun()

# ==========================================
# 🎥 THE AI CAMERA LOOP
# ==========================================
if st.session_state.camera_active:

    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(CAMERA_SOURCE)
        st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    sequence_buffer = []
    exercise_state = "up"
    reps = 0
    frame_count = 0
    current_angles_display = {}

    last_spoken_status = ""
    last_spoken_time = 0

    rep_config = exercise["rep_detection"]
    feedback_rules = exercise["feedback_rules"]
    target = st.session_state.target_angles.get(selected_key, cal_config["default_target"])

    while (st.session_state.cap is not None
           and st.session_state.cap.isOpened()
           and st.session_state.camera_active):

        success, frame = st.session_state.cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 2 != 0:
            continue

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status_text = "WAITING..."
        status_class = "status-idle"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Visibility check
            required = exercise["required_landmarks"]
            vis_scores = [lm[idx].visibility for idx in required]

            nose_y = lm[0].y
            ankle_y = (lm[27].y + lm[28].y) / 2.0
            body_height = abs(ankle_y - nose_y)

            if min(vis_scores) < 0.65 or body_height < 0.35:
                status_text = "STEP INTO FRAME"
                status_class = "status-warn"
                sequence_buffer.clear()
                exercise_state = "up"
            else:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Compute all angles
                angles = compute_all_angles(lm)
                current_angles_display = angles

                # Build feature vector
                feature_vector = build_feature_vector(lm, angles)
                sequence_buffer.append(feature_vector)
                sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

                # ==========================================
                # 🔄 REP DETECTION
                # ==========================================
                if selected_key == "glute_bridge":
                    hip_angle = angles.get("avg_hip", 0)

                    if hip_angle > rep_config.get("top_hip", 165):
                        if exercise_state == "down":
                            reps += 1
                            exercise_state = "up"
                            speak(f"{reps}")
                    elif hip_angle < rep_config.get("down_hip", 130):
                        exercise_state = "down"

                    # Form feedback
                    if hip_angle > rep_config.get("down_hip", 130):
                        feedback_found = False
                        for angle_name, condition, threshold, message, severity in feedback_rules:
                            angle_val = angles.get(angle_name, 180)
                            actual_threshold = threshold if threshold is not None else target

                            if condition == "below" and angle_val < actual_threshold:
                                status_text = message
                                status_class = "status-warn"
                                feedback_found = True
                                break
                            elif condition == "above" and angle_val > actual_threshold:
                                status_text = message
                                status_class = "status-warn"
                                feedback_found = True
                                break

                        if not feedback_found:
                            status_text = exercise["perfect_message"]
                            status_class = "status-good"
                    else:
                        status_text = "IDLE"
                        status_class = "status-idle"

                else:
                    # Squats / Lunges
                    up_knee = angles.get(rep_config["angle_for_up_knee"], 180)
                    up_hip = angles.get(rep_config["angle_for_up_hip"], 180)

                    up_thresh_knee = rep_config.get("up_knee", 160)
                    up_thresh_hip = rep_config.get("up_hip", 155)

                    if up_thresh_knee and up_thresh_hip and up_knee > up_thresh_knee and up_hip > up_thresh_hip:
                        status_text = "IDLE"
                        status_class = "status-idle"
                        if exercise_state == "down":
                            reps += 1
                            exercise_state = "up"
                            speak(f"{reps}")
                    else:
                        down_angle = angles.get(rep_config["angle_for_down"], 180)
                        down_offset = rep_config.get("down_knee_offset", 15)
                        if down_angle <= (target + down_offset):
                            exercise_state = "down"

                        # Form feedback
                        feedback_found = False
                        for angle_name, condition, threshold, message, severity in feedback_rules:
                            angle_val = angles.get(angle_name, 180)
                            actual_threshold = threshold if threshold is not None else (target + 15)

                            if condition == "below" and angle_val < actual_threshold:
                                status_text = message
                                status_class = "status-warn"
                                feedback_found = True
                                break
                            elif condition == "above" and angle_val > actual_threshold:
                                status_text = message
                                status_class = "status-warn"
                                feedback_found = True
                                break

                        if not feedback_found:
                            status_text = exercise["perfect_message"]
                            status_class = "status-good"

        # ==========================================
        # 🗣️ SMART VOICE COACH
        # ==========================================
        current_time = time.time()
        if status_text not in ["WAITING...", "STEP INTO FRAME", "IDLE"]:
            if status_text != last_spoken_status or (current_time - last_spoken_time > 3.0):
                speak(status_text)
                last_spoken_status = status_text
                last_spoken_time = current_time

        if status_text in ["IDLE", "WAITING...", "STEP INTO FRAME"]:
            last_spoken_status = ""

        # ==========================================
        # 🖼️ UPDATE UI
        # ==========================================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        status_placeholder.markdown(f"""
            <div style="text-align:center; padding: 12px; background: linear-gradient(135deg, #1e293b, #334155);
                        border-radius: 12px; border: 1px solid #475569; margin-top: 8px;">
                <span class="{status_class}">{exercise['name']}: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)

        reps_placeholder.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">REPETITIONS</p>
                <p class="metric-value">{reps}</p>
            </div>
        """, unsafe_allow_html=True)

        # Update angle cards dynamically
        for label, angle_key in exercise["display_angles"].items():
            angle_val = current_angles_display.get(angle_key, 0)
            if label in angle_placeholders:
                angle_placeholders[label].markdown(f"""
                    <div class="angle-card">
                        <p class="angle-name">{label}</p>
                        <p class="angle-val">{int(angle_val)}°</p>
                    </div>
                """, unsafe_allow_html=True)