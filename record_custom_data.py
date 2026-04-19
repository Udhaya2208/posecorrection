import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
import os

EXPECTED_FEATURES = 240

def calculate_angle(a, b, c):
    """Calculates the angle between 3 points"""
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def main():
    print("=======================================")
    print("   POSEGURU: HYBRID DATA RECORDER      ")
    print("=======================================")
    print("0 - Record CORRECT Squats")
    print("1 - Record INCORRECT Squats")
    print("2 - Record IDLE (Standing still)")
    
    label = input("Enter the number for what you are recording: ")
    
    if label == '0': filename = "Data_Correct.csv"
    elif label == '1': filename = "Data_Incorrect.csv"
    elif label == '2': filename = "Data_Idle.csv"
    else: print("Invalid input!"); return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    data_list = []
    
    print("Get ready! Recording starts in 5 seconds...")
    cv2.waitKey(5000)
    print("🔴 RECORDING NOW! (Do the movement for 30 seconds)")
    
    start_time = time.time()

    while time.time() - start_time < 30: # Record for 30 seconds
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            
            # 1. Get raw coordinates
            features = []
            for landmark in lm: 
                features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # 2. 📐 ADD THE ANGLES TO THE TRAINING DATA!
            knee_angle = (calculate_angle(lm[23], lm[25], lm[27]) + calculate_angle(lm[24], lm[26], lm[28])) / 2.0
            hip_angle = (calculate_angle(lm[11], lm[23], lm[25]) + calculate_angle(lm[12], lm[24], lm[26])) / 2.0
            features.extend([knee_angle, hip_angle])
            
            # 3. Pad to 240
            features.extend([0.0] * (EXPECTED_FEATURES - len(features)))
            data_list.append(features)

        cv2.putText(frame, f"RECORDING: {filename}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Recording Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    df = pd.DataFrame(data_list)
    # If file exists, append to it. If not, create it.
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
        
    print(f"✅ Successfully saved {len(data_list)} frames to {filename} with Angle Data!")

if __name__ == "__main__":
    main()