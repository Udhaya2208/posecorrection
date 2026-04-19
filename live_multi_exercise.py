import cv2
import numpy as np
import mediapipe as mp
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from exercise_config import (
    EXERCISES, EXPECTED_FEATURES, SEQUENCE_LENGTH,
    BRANCH_INDICES, ANGLE_FEATURE_INDICES,
    compute_all_angles, build_feature_vector, get_exercise_list
)

# ==========================================
# 🏋️ POSEGURU: LIVE MULTI-EXERCISE CORRECTION
# ==========================================
# User SELECTS which exercise to perform.
# Loads the corresponding specialist model.
# Provides real-time form correction + rep counting.
# ==========================================

CALIBRATION_FILE = "user_calibration.json"
OUTPUT_VIDEO_NAME = "Actionable_Rehab_FYP.avi"

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
# 🎯 LOAD CALIBRATION DATA
# ==========================================
def load_calibration():
    """Load calibration targets from JSON file."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return {}


# ==========================================
# 🚀 MAIN APPLICATION
# ==========================================
def main():
    print("=" * 55)
    print("   POSEGURU: LIVE MULTI-EXERCISE CORRECTION")
    print("=" * 55)

    # ==========================================
    # 🎯 STEP 1: SELECT EXERCISE
    # ==========================================
    exercise_keys = get_exercise_list()
    print("\n📋 Select Exercise:")
    for i, key in enumerate(exercise_keys):
        ex = EXERCISES[key]
        model_file = f"{ex['csv_prefix'].lower()}_specialist.h5"
        exists = "✅" if os.path.exists(model_file) else "❌ (no model)"
        print(f"  {i + 1} - {ex['display_name']}  {exists}")

    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(exercise_keys):
                break
            print("Invalid number!")
        except ValueError:
            print("Please enter a number!")

    selected_key = exercise_keys[choice]
    exercise = EXERCISES[selected_key]

    print(f"\n✅ Selected: {exercise['display_name']}")

    # ==========================================
    # 🧠 STEP 2: LOAD AI MODEL
    # ==========================================
    model_file = f"{exercise['csv_prefix'].lower()}_specialist.h5"
    model = None

    if os.path.exists(model_file):
        print(f"⏳ Loading model: {model_file}...")
        model = load_model(model_file, custom_objects={
            'get_branch_features': get_branch_features,
            'get_angle_features': get_angle_features,
            'tf': tf
        })
        print("✅ Model loaded!")
    else:
        print(f"⚠️ Model not found: {model_file}")
        print("   Running in ANGLE-ONLY mode (no AI classification)")

    # ==========================================
    # 🎯 STEP 3: LOAD CALIBRATION
    # ==========================================
    cal_data = load_calibration()
    cal_config = exercise["calibration"]

    if selected_key in cal_data:
        custom_target = float(cal_data[selected_key])
        print(f"✅ Calibration loaded: Target = {custom_target}°")
    else:
        custom_target = float(cal_config["default_target"])
        print(f"⚠️ No calibration found. Using default: {custom_target}°")

    # ==========================================
    # 🎥 STEP 4: INITIALIZE CAMERA
    # ==========================================
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30,
                          (int(cap.get(3)), int(cap.get(4))))

    # ==========================================
    # 🔄 STEP 5: STATE VARIABLES
    # ==========================================
    sequence_buffer = []
    exercise_state = "up"  # or "down" for rep tracking
    reps = 0
    status = "WAITING..."
    color = (0, 255, 255)
    frame_count = 0
    class_id = 2  # Default: Idle

    rep_config = exercise["rep_detection"]
    feedback_rules = exercise["feedback_rules"]

    print(f"\n🎬 LIVE TRACKING STARTED! Exercise: {exercise['name']}")
    print(f"   Press 'q' to quit.\n")

    # ==========================================
    # 🎬 STEP 6: MAIN LOOP
    # ==========================================
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check visibility of required landmarks
            required = exercise["required_landmarks"]
            visibility_ok = all(lm[idx].visibility > 0.5 for idx in required)

            if not visibility_ok:
                status = "STEP INTO FRAME"
                color = (0, 0, 255)
                sequence_buffer.clear()
            else:
                # Compute all angles
                angles = compute_all_angles(lm)

                # Build feature vector
                feature_vector = build_feature_vector(lm, angles)
                sequence_buffer.append(feature_vector)
                sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

                # AI prediction (every 4 frames)
                if model and len(sequence_buffer) == SEQUENCE_LENGTH and frame_count % 4 == 0:
                    class_id = np.argmax(
                        model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0)
                    )

                # ==========================================
                # 🔄 REP DETECTION (exercise-specific)
                # ==========================================
                if selected_key == "glute_bridge":
                    # Glute bridge: reps counted at top of bridge
                    hip_angle = angles.get("avg_hip", 0)

                    if hip_angle > rep_config.get("top_hip", 165):
                        if exercise_state == "down":
                            reps += 1
                            exercise_state = "up"
                    elif hip_angle < rep_config.get("down_hip", 130):
                        exercise_state = "down"
                else:
                    # Squats / Lunges: standard up/down detection
                    up_knee_angle = angles.get(rep_config["angle_for_up_knee"], 180)
                    up_hip_angle = angles.get(rep_config["angle_for_up_hip"], 180)
                    down_angle = angles.get(rep_config["angle_for_down"], 180)

                    up_threshold_knee = rep_config.get("up_knee", 160)
                    up_threshold_hip = rep_config.get("up_hip", 155)

                    if up_threshold_knee and up_threshold_hip:
                        if up_knee_angle > up_threshold_knee and up_hip_angle > up_threshold_hip:
                            status = "IDLE"
                            color = (0, 255, 255)
                            if exercise_state == "down":
                                reps += 1
                                exercise_state = "up"
                        else:
                            # Mark as going down
                            down_offset = rep_config.get("down_knee_offset", 15)
                            if down_angle <= (custom_target + down_offset):
                                exercise_state = "down"

                # ==========================================
                # 🛑 FORM FEEDBACK (exercise-specific rules)
                # ==========================================
                if selected_key == "glute_bridge":
                    hip_angle = angles.get("avg_hip", 0)
                    if hip_angle > rep_config.get("down_hip", 130):
                        # User is in active bridge position, check form
                        feedback_found = False
                        for angle_name, condition, threshold, message, severity in feedback_rules:
                            angle_val = angles.get(angle_name, 180)

                            # Handle dynamic threshold (None = use calibration target)
                            actual_threshold = threshold if threshold is not None else custom_target

                            if condition == "below" and angle_val < actual_threshold:
                                status = message
                                color = (0, 0, 255) if severity == "danger" else (0, 165, 255)
                                feedback_found = True
                                break
                            elif condition == "above" and angle_val > actual_threshold:
                                status = message
                                color = (0, 0, 255) if severity == "danger" else (0, 165, 255)
                                feedback_found = True
                                break

                        if not feedback_found:
                            status = exercise["perfect_message"]
                            color = (0, 255, 0)
                    else:
                        status = "IDLE"
                        color = (0, 255, 255)

                else:
                    # Squats / Lunges
                    up_knee_angle = angles.get(rep_config["angle_for_up_knee"], 180)
                    up_hip_angle = angles.get(rep_config["angle_for_up_hip"], 180)
                    up_threshold_knee = rep_config.get("up_knee", 160)
                    up_threshold_hip = rep_config.get("up_hip", 155)

                    if up_threshold_knee and up_threshold_hip:
                        if not (up_knee_angle > up_threshold_knee and up_hip_angle > up_threshold_hip):
                            # User is in active movement — apply feedback rules
                            feedback_found = False
                            for angle_name, condition, threshold, message, severity in feedback_rules:
                                angle_val = angles.get(angle_name, 180)
                                actual_threshold = threshold if threshold is not None else (custom_target + 15)

                                if condition == "below" and angle_val < actual_threshold:
                                    status = message
                                    color = (0, 0, 255) if severity == "danger" else (0, 165, 255)
                                    feedback_found = True
                                    break
                                elif condition == "above" and angle_val > actual_threshold:
                                    status = message
                                    color = (0, 0, 255) if severity == "danger" else (0, 165, 255)
                                    feedback_found = True
                                    break

                            if not feedback_found:
                                status = exercise["perfect_message"]
                                color = (0, 255, 0)

                # ==========================================
                # 📊 DRAW ANGLE VALUES ON SCREEN
                # ==========================================
                y_pos = 180
                for angle_label, angle_key in exercise["display_angles"].items():
                    angle_val = angles.get(angle_key, 0)
                    cv2.putText(frame, f"{angle_label}: {int(angle_val)} deg",
                                (420, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_pos += 30

        # ==========================================
        # 🎨 UI OVERLAY
        # ==========================================
        cv2.rectangle(frame, (0, 0), (640, 70), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, f"{exercise['name']}: {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"REPS: {reps}  |  Goal: {int(custom_target)} deg", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow(f"PoseGuru: {exercise['name']}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ==========================================
    # 🧹 CLEANUP
    # ==========================================
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n🎉 Session Complete!")
    print(f"   Exercise: {exercise['display_name']}")
    print(f"   Reps: {reps}")
    print(f"   Video saved: {OUTPUT_VIDEO_NAME}")


if __name__ == "__main__":
    main()
