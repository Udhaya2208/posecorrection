import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
import os
from exercise_config import (
    EXERCISES, EXPECTED_FEATURES, compute_all_angles,
    build_feature_vector, get_exercise_list
)

# ==========================================
# 🏋️ POSEGURU: MULTI-EXERCISE DATA RECORDER
# ==========================================
# Records training data for Squats, Lunges, and Glute Bridges.
# Saves 248-feature vectors (132 landmarks + 8 angles + padding).
# ==========================================

def main():
    print("=" * 50)
    print("   POSEGURU: MULTI-EXERCISE DATA RECORDER")
    print("=" * 50)

    # ==========================================
    # 🎯 STEP 1: SELECT EXERCISE
    # ==========================================
    exercise_keys = get_exercise_list()
    print("\n📋 Select Exercise:")
    for i, key in enumerate(exercise_keys):
        ex = EXERCISES[key]
        print(f"  {i + 1} - {ex['display_name']}  ({ex['description'][:50]}...)")

    while True:
        try:
            choice = int(input("\nEnter exercise number: ")) - 1
            if 0 <= choice < len(exercise_keys):
                selected_exercise = exercise_keys[choice]
                break
            print("Invalid number! Try again.")
        except ValueError:
            print("Please enter a number!")

    exercise = EXERCISES[selected_exercise]
    print(f"\n✅ Selected: {exercise['display_name']}")

    # ==========================================
    # 🎯 STEP 2: SELECT QUALITY LABEL
    # ==========================================
    print("\n📋 Select Recording Type:")
    print("  0 - Record CORRECT form")
    print("  1 - Record INCORRECT form")
    print("  2 - Record IDLE (standing still)")

    while True:
        label = input("\nEnter the number: ").strip()
        if label in ['0', '1', '2']:
            break
        print("Invalid! Enter 0, 1, or 2.")

    label_names = {
        '0': 'Correct',
        '1': 'Incorrect',
        '2': 'Idle'
    }

    # Create filename: e.g., Squat_Correct.csv, Lunge_Incorrect.csv
    filename = f"{exercise['csv_prefix']}_{label_names[label]}.csv"
    quality_text = label_names[label].upper()

    print(f"\n🎯 Recording: {exercise['display_name']} - {quality_text}")
    print(f"📁 Output File: {filename}")

    # ==========================================
    # 🎥 STEP 3: INITIALIZE CAMERA & MEDIAPIPE
    # ==========================================
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return

    data_list = []

    # ==========================================
    # ⏳ STEP 4: COUNTDOWN
    # ==========================================
    print("\n⏳ Get into position! Recording starts in 5 seconds...")
    print(f"   📐 Tracked angles: {', '.join(exercise['primary_angles'])}")

    countdown_start = time.time()
    while time.time() - countdown_start < 5:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            remaining = 5 - int(time.time() - countdown_start)
            cv2.putText(frame, f"GET READY: {remaining}s", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(frame, f"{exercise['name']} - {quality_text}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Recording Data", frame)
            cv2.waitKey(1)

    print("🔴 RECORDING NOW! Perform the movement for 30 seconds.")
    print("   Press 'q' to stop early.\n")

    # ==========================================
    # 🎬 STEP 5: RECORDING LOOP (30 seconds)
    # ==========================================
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 30:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # 1. Compute all 8 angles
            angles = compute_all_angles(lm)

            # 2. Build the full 248-feature vector
            feature_vector = build_feature_vector(lm, angles)
            data_list.append(feature_vector)
            frame_count += 1

            # 3. Display live angle values on screen
            y_offset = 200
            for angle_label, angle_key in exercise["display_angles"].items():
                angle_val = angles.get(angle_key, 0)
                cv2.putText(frame, f"{angle_label}: {int(angle_val)} deg", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35

        # UI overlay
        elapsed = int(time.time() - start_time)
        remaining = 30 - elapsed
        color = (0, 0, 255) if label == '1' else (0, 255, 0) if label == '0' else (255, 255, 0)

        cv2.rectangle(frame, (0, 0), (640, 60), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, f"REC: {exercise['name']} {quality_text} | {remaining}s | {frame_count} frames",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Recording Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ==========================================
    # 💾 STEP 6: SAVE DATA
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()

    if len(data_list) == 0:
        print("❌ No data recorded! Make sure your body is visible to the camera.")
        return

    df = pd.DataFrame(data_list)

    # If file exists, append to it
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"\n📎 Appended {len(data_list)} frames to existing {filename}")
    else:
        df.to_csv(filename, index=False)
        print(f"\n📁 Created new {filename}")

    print(f"✅ Successfully saved {len(data_list)} frames with 8-angle feature data!")
    print(f"   Feature vector size: {EXPECTED_FEATURES} per frame")
    print(f"   Angles tracked: {', '.join(exercise['primary_angles'] + exercise.get('secondary_angles', []))}")

    # ==========================================
    # 📊 STEP 7: DATA SUMMARY
    # ==========================================
    print("\n" + "=" * 50)
    print("📊 COLLECTION PROGRESS:")
    for ex_key, ex_data in EXERCISES.items():
        prefix = ex_data["csv_prefix"]
        for quality in ["Correct", "Incorrect", "Idle"]:
            fname = f"{prefix}_{quality}.csv"
            if os.path.exists(fname):
                rows = len(pd.read_csv(fname))
                status = "✅" if rows > 200 else "⚠️"
                print(f"   {status} {fname}: {rows} frames")
            else:
                print(f"   ❌ {fname}: NOT YET RECORDED")
    print("=" * 50)


if __name__ == "__main__":
    main()
