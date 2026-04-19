import cv2
import mediapipe as mp
import csv
import time
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
OUTPUT_CSV = "pose_data.csv"
OUTPUT_VIDEO = "output_with_landmarks.avi"
IMAGE_FOLDER = "captured_frames"

# Create image folder if it doesn't exist
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Camera
cap = cv2.VideoCapture(0)

# ==========================================
# 🎥 SETUP VIDEO WRITER
# ==========================================
# Get camera dimensions so the video file matches perfectly
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20.0

# Create VideoWriter (Saves the video with landmarks)
out = cv2.VideoWriter(OUTPUT_VIDEO, 
                      cv2.VideoWriter_fourcc('M','J','P','G'), 
                      fps, 
                      (frame_width, frame_height))

# ==========================================
# 📝 PREPARE CSV FILE
# ==========================================
header = ["frame_id", "timestamp"]
for i in range(33):
    header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_vis"])

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"✅ Recording Started!")
print(f"   1. Raw Images -> {IMAGE_FOLDER}/")
print(f"   2. Data -> {OUTPUT_CSV}")
print(f"   3. Video (Marked) -> {OUTPUT_VIDEO}")
print("Press 'q' to stop.")

# ==========================================
# 🎬 MAIN LOOP
# ==========================================
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # 1. SAVE RAW IMAGE (Clean, for training)
    # We save this BEFORE drawing lines on it
    image_filename = os.path.join(IMAGE_FOLDER, f"frame_{frame_count}.jpg")
    cv2.imwrite(image_filename, frame)

    # 2. POSE ESTIMATION
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 3. DRAW LANDMARKS & SAVE VIDEO + CSV
    if results.pose_landmarks:
        # A. Save Data to CSV
        row = [frame_count, time.time() - start_time]
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # B. Draw Landmarks on the Frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # C. Write the Marked Frame to Video File
        out.write(frame)

    # 4. SHOW ON SCREEN
    cv2.putText(frame, f"REC: {frame_count}", (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Full Data Collector', frame)
    frame_count += 1
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ All data saved successfully.")
