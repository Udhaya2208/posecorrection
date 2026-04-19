import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import sys
from exercise_config import (
    EXERCISES, compute_all_angles, calculate_angle, get_exercise_list
)

# ==========================================
# 🎯 POSEGURU: MULTI-EXERCISE CALIBRATION
# ==========================================
# Calibrates the user's safe range of motion
# for each exercise SEPARATELY.
# Saves to user_calibration.json
# ==========================================

CALIBRATION_FILE = "user_calibration.json"


def load_calibration():
    """Load existing calibration data or return empty dict."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return {}


def save_calibration(data):
    """Save calibration data to JSON."""
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def calibrate_exercise(exercise_key):
    """
    Run calibration for a specific exercise.
    Records the target angle for 10 seconds and saves the safe limit.
    """
    exercise = EXERCISES[exercise_key]
    cal_config = exercise["calibration"]
    target_angle_name = cal_config["angle"]

    print(f"\n{'=' * 50}")
    print(f"  CALIBRATING: {exercise['display_name']}")
    print(f"{'=' * 50}")
    print(f"\n📐 Tracking: {target_angle_name}")
    print(f"📝 Instruction: {cal_config['instruction']}")
    print(f"⏱️  You have 20 seconds total (10s countdown + 10s recording)")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return None

    start_time = time.time()

    # For squats/lunges: track the LOWEST angle (deepest bend)
    # For glute bridges: track the HIGHEST angle (maximum extension)
    if cal_config["offset"] > 0:
        # Positive offset → we track lowest (squats, lunges)
        best_angle = 180.0
        compare = lambda new, old: new < old
    else:
        # Negative offset → we track highest (glute bridges)
        best_angle = 0.0
        compare = lambda new, old: new > old

    recording_started = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time
        frame_count += 1

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Compute all angles
            angles = compute_all_angles(lm)
            current_angle = angles.get(target_angle_name, 180.0)

            if elapsed < 10:
                # COUNTDOWN PHASE
                time_left = int(10 - elapsed)
                cv2.putText(frame, f"GET READY: {time_left}s",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, cal_config["instruction"],
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # RECORDING PHASE
                if not recording_started:
                    recording_started = True
                    print("\n🔴 RECORDING! Do the movement now!")

                if compare(current_angle, best_angle):
                    best_angle = current_angle

                cv2.putText(frame, f"CALIBRATING: {exercise['name']}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f"Current: {int(current_angle)} deg",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Best: {int(best_angle)} deg",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Always show current angle
            cv2.putText(frame, f"{target_angle_name}: {int(current_angle)}",
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(f'Calibration - {exercise["name"]}', frame)

        # Stop after 20 seconds total or on 'q'
        if elapsed > 20 or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate target with offset
    if best_angle < 170 or cal_config["offset"] < 0:
        target = int(best_angle + cal_config["offset"])
        print(f"\n✅ Calibration Complete!")
        print(f"   Best recorded: {int(best_angle)}°")
        print(f"   Offset applied: {cal_config['offset']:+d}°")
        print(f"   Target saved: {target}°")
        return target
    else:
        print(f"\n⚠️ No significant movement detected. Using default: {cal_config['default_target']}°")
        return cal_config["default_target"]


def main():
    print("=" * 50)
    print("   POSEGURU: MULTI-EXERCISE CALIBRATION")
    print("=" * 50)

    # Check for command-line argument
    if len(sys.argv) > 1:
        # Called with: python calibrate_user.py --exercise squat
        exercise_key = None
        for i, arg in enumerate(sys.argv):
            if arg == "--exercise" and i + 1 < len(sys.argv):
                exercise_key = sys.argv[i + 1]
                break

        if exercise_key and exercise_key in EXERCISES:
            target = calibrate_exercise(exercise_key)
            if target is not None:
                cal_data = load_calibration()
                cal_data[exercise_key] = target
                save_calibration(cal_data)
                print(f"\n💾 Saved to {CALIBRATION_FILE}")
            return

    # Interactive mode
    exercise_keys = get_exercise_list()

    # Show current calibration status
    cal_data = load_calibration()
    print("\n📊 Current Calibration Status:")
    for key in exercise_keys:
        if key in cal_data:
            print(f"   ✅ {EXERCISES[key]['display_name']}: {cal_data[key]}°")
        else:
            print(f"   ❌ {EXERCISES[key]['display_name']}: Not calibrated (using default)")

    print("\n📋 Select Exercise to Calibrate:")
    for i, key in enumerate(exercise_keys):
        print(f"  {i + 1} - {EXERCISES[key]['display_name']}")
    print(f"  {len(exercise_keys) + 1} - Calibrate ALL exercises")

    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice <= len(exercise_keys):
                break
            print("Invalid number!")
        except ValueError:
            print("Please enter a number!")

    # Calibrate selected exercise(s)
    if choice == len(exercise_keys):
        exercises_to_calibrate = exercise_keys
    else:
        exercises_to_calibrate = [exercise_keys[choice]]

    for ex_key in exercises_to_calibrate:
        target = calibrate_exercise(ex_key)
        if target is not None:
            cal_data[ex_key] = target

    # Save all calibrations
    save_calibration(cal_data)
    print(f"\n💾 All calibrations saved to {CALIBRATION_FILE}")

    # Final summary
    print(f"\n{'=' * 50}")
    print("📊 CALIBRATION SUMMARY:")
    for key in exercise_keys:
        if key in cal_data:
            print(f"   ✅ {EXERCISES[key]['display_name']}: {cal_data[key]}°")
        else:
            print(f"   ❌ {EXERCISES[key]['display_name']}: Not calibrated")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()