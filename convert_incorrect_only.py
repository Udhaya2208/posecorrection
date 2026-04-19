import os
import numpy as np
import csv

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Pointing exactly to your incorrect data folder
UIPRMD_FOLDER = r"D:\PoseGuru_Project1\incorrect" 

EXPECTED_FEATURES = 240

def main():
    print("=========================================")
    print("   POSEGURU: INCORRECT DATA EXTRACTOR    ")
    print("=========================================")
    
    if not os.path.exists(UIPRMD_FOLDER):
        print(f"❌ ERROR: Folder not found at {UIPRMD_FOLDER}")
        return

    output_filename = "Data_Incorrect.csv"
    file_incorrect = open(output_filename, mode='w', newline='')
    writer_incorrect = csv.writer(file_incorrect)
    
    # Write the 240 headers
    headers = [f'feature_{i}' for i in range(EXPECTED_FEATURES)]
    writer_incorrect.writerow(headers)
    
    files = os.listdir(UIPRMD_FOLDER)
    incorrect_count = 0

    print("⏳ Scanning for INCORRECT (_inc) files...")

    for file in files:
        if "m01" in file and "inc" in file:
            
            file_path = os.path.join(UIPRMD_FOLDER, file)
            
            try:
                # FIX: Removed delimiter=',' so it splits by Tabs/Spaces automatically!
                data = np.loadtxt(file_path)
                
                # Safety check: if the file only has 1 row, format it properly
                if data.ndim == 1:
                    data = np.expand_dims(data, axis=0)
                
                for row in data:
                    features = row.tolist()
                    
                    # Pad with zeros to reach exactly 240 columns
                    features.extend([0.0] * (EXPECTED_FEATURES - len(features)))
                    
                    # Write to the CSV file
                    writer_incorrect.writerow(features)

                incorrect_count += 1
                
            except Exception as e:
                print(f"⚠️ Skipping {file} due to error: {e}")

    # Close the file safely
    file_incorrect.close()

    print("\n✅ EXTRACTION COMPLETE!")
    print(f"Processed {incorrect_count} Incorrect videos.")
    print(f"Your '{output_filename}' is ready!")

if __name__ == "__main__":
    main()