import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "squat_specialist_v3.h5"
CALIBRATION_FILE = "user_calibration.txt" 
SEQUENCE_LENGTH = 30
EXPECTED_FEATURES = 240
OUTPUT_VIDEO_NAME = "Actionable_Rehab_FYP.avi"

# ==========================================
# 🧩 CUSTOM AI FUNCTIONS
# ==========================================
def get_branch_features(x, landmark_indices):
    feature_indices = []
    for i in landmark_indices:
        feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
    return tf.gather(x, feature_indices, axis=2)

def get_angle_features(x):
    return tf.gather(x, [132, 133], axis=2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================
# 🚀 MAIN APPLICATION
# ==========================================
def main():
    print("⏳ Loading Fast & Smart AI Model...")
    model = load_model(MODEL_PATH, custom_objects={
        'get_branch_features': get_branch_features,
        'get_angle_features': get_angle_features,
        'tf': tf
    })
    
    # 🎯 LOAD CALIBRATION (Default to 120 if file missing)
    custom_target = 120.0 
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            custom_target = float(f.read().strip())
        print(f"✅ Calibration Loaded: Target Depth {custom_target}°")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    sequence_buffer = []
    squat_state = "up"; reps = 0
    status = "Waiting..."; color = (0, 255, 255)
    frame_count = 0; class_id = 2 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if min([lm[23].visibility, lm[24].visibility, lm[25].visibility, lm[26].visibility]) < 0.5:
                status = "BACK UP (SHOW LEGS)"; color = (0, 0, 255); sequence_buffer.clear()
            else:
                features = []
                for landmark in lm: features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # 📐 MATH CALCULATION
                current_knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
                current_hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
                
                features.extend([current_knee_angle, current_hip_angle])
                features.extend([0.0] * (EXPECTED_FEATURES - len(features)))
                sequence_buffer.append(features)
                sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    if frame_count % 4 == 0:
                        class_id = np.argmax(model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0))

                    # 🚀 THE STAND-UP RESET (Hysteresis)
                    if current_knee_angle > 170 and current_hip_angle > 165:
                        status = "IDLE"; color = (0, 255, 255)
                        if squat_state == "down": reps += 1; squat_state = "up"
                    
                    # 🛑 THE INJURY-ADAPTIVE SAFETY NET
                    else:
                        squat_state = "down"
                        
                        # 1. Check Back Safety (Stops leaning too far forward)
                        if current_hip_angle < 100: 
                            status = "STRAIGHTEN YOUR BACK"; color = (0, 0, 255)
                        
                        # 2. Check Depth based on INDIVIDUAL CALIBRATION
                        # If knee is more than 5 degrees above their personal target
                        elif current_knee_angle > (custom_target + 5):
                            status = "GO DOWN SLIGHTLY"; color = (0, 0, 255)
                        
                        # 3. User has reached their personal goal!
                        else:
                            if custom_target >= 150:
                                status = "GOOD (ADAPTED REHAB)"; color = (0, 255, 0)
                            else:
                                status = "PERFECT SQUAT!"; color = (0, 255, 0)

        # UI DRAWING
        cv2.rectangle(frame, (0, 0), (640, 70), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"REPS: {reps}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        if 'current_knee_angle' in locals():
            cv2.putText(frame, f"Knee: {int(current_knee_angle)}", (450, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Goal: {int(custom_target)}", (450, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('PoseGuru: Spatio-Temporal AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()