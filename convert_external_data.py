import os
import numpy as np
import csv
from exercise_config import EXERCISES, EXPECTED_FEATURES

# ==========================================
# 🔄 POSEGURU: EXTERNAL DATA CONVERTER
# ==========================================
# Converts external motion-capture datasets (e.g., UI-PRMD)
# into the PoseGuru 248-feature CSV format for any exercise.
# ==========================================

# ==========================================
# ⚙️ CONFIGURATION: Set your data folders here
# ==========================================
EXTERNAL_DATA_PATHS = {
    "squat": {
        "incorrect": r"D:\PoseGuru_Project1\incorrect",
        "correct": r"D:\PoseGuru_Project1\correct",
    },
    "lunge": {
        "incorrect": r"D:\PoseGuru_Project1\lunge_incorrect",
        "correct": r"D:\PoseGuru_Project1\lunge_correct",
    },
    "glute_bridge": {
        "incorrect": r"D:\PoseGuru_Project1\bridge_incorrect",
        "correct": r"D:\PoseGuru_Project1\bridge_correct",
    },
}

# File pattern filters for UI-PRMD dataset
# e.g., "m01" = movement 01 (deep squat), "inc" = incorrect
FILE_PATTERNS = {
    "squat": {
        "movement_id": "m01",
        "incorrect_tag": "inc",
        "correct_tag": "cor",
    },
    "lunge": {
        "movement_id": "m05",    # Adjust based on your dataset
        "incorrect_tag": "inc",
        "correct_tag": "cor",
    },
    "glute_bridge": {
        "movement_id": "m08",    # Adjust based on your dataset
        "incorrect_tag": "inc",
        "correct_tag": "cor",
    },
}


def convert_folder(folder_path, output_csv, movement_id, quality_tag):
    """
    Convert all matching files in a folder to PoseGuru CSV format.
    """
    if not os.path.exists(folder_path):
        print(f"   ⚠️ Folder not found: {folder_path}")
        return 0

    file_out = open(output_csv, mode='w', newline='')
    writer = csv.writer(file_out)

    # Write headers
    headers = [f'feature_{i}' for i in range(EXPECTED_FEATURES)]
    writer.writerow(headers)

    files = os.listdir(folder_path)
    converted_count = 0
    total_frames = 0

    for file in sorted(files):
        # Filter by movement ID and quality tag
        if movement_id in file and quality_tag in file:
            file_path = os.path.join(folder_path, file)

            try:
                # Load data (auto-detect delimiter: tabs or spaces)
                data = np.loadtxt(file_path)

                # Handle single-row files
                if data.ndim == 1:
                    data = np.expand_dims(data, axis=0)

                for row in data:
                    features = row.tolist()

                    # Pad with zeros to reach exactly EXPECTED_FEATURES columns
                    if len(features) < EXPECTED_FEATURES:
                        features.extend([0.0] * (EXPECTED_FEATURES - len(features)))
                    elif len(features) > EXPECTED_FEATURES:
                        features = features[:EXPECTED_FEATURES]

                    writer.writerow(features)
                    total_frames += 1

                converted_count += 1

            except Exception as e:
                print(f"   ⚠️ Skipping {file}: {e}")

    file_out.close()
    return converted_count, total_frames


def main():
    print("=" * 55)
    print("   POSEGURU: MULTI-EXERCISE DATA CONVERTER")
    print("=" * 55)
    print("\nConverts external motion-capture data to PoseGuru format.\n")

    # ==========================================
    # 🎯 SELECT EXERCISE
    # ==========================================
    exercise_keys = list(EXERCISES.keys())
    print("📋 Select Exercise to Convert:")
    for i, key in enumerate(exercise_keys):
        print(f"  {i + 1} - {EXERCISES[key]['display_name']}")
    print(f"  {len(exercise_keys) + 1} - Convert ALL exercises")

    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice <= len(exercise_keys):
                break
            print("Invalid number!")
        except ValueError:
            print("Please enter a number!")

    # Determine which exercises to convert
    if choice == len(exercise_keys):
        exercises_to_convert = exercise_keys
    else:
        exercises_to_convert = [exercise_keys[choice]]

    # ==========================================
    # 🔄 CONVERT DATA
    # ==========================================
    for ex_key in exercises_to_convert:
        exercise = EXERCISES[ex_key]
        patterns = FILE_PATTERNS.get(ex_key, {})
        paths = EXTERNAL_DATA_PATHS.get(ex_key, {})

        print(f"\n{'=' * 40}")
        print(f"  Converting: {exercise['display_name']}")
        print(f"{'=' * 40}")

        # Convert Incorrect Data
        if "incorrect" in paths:
            output_csv = f"{exercise['csv_prefix']}_Incorrect.csv"
            print(f"\n⏳ Converting INCORRECT data...")
            print(f"   Source: {paths['incorrect']}")
            print(f"   Output: {output_csv}")

            result = convert_folder(
                paths["incorrect"],
                output_csv,
                patterns.get("movement_id", ""),
                patterns.get("incorrect_tag", "inc")
            )

            if isinstance(result, tuple):
                files_count, frames_count = result
                print(f"   ✅ Converted {files_count} files → {frames_count} frames")
            else:
                print(f"   ⚠️ No data converted")

        # Convert Correct Data
        if "correct" in paths:
            output_csv = f"{exercise['csv_prefix']}_Correct.csv"
            print(f"\n⏳ Converting CORRECT data...")
            print(f"   Source: {paths['correct']}")
            print(f"   Output: {output_csv}")

            result = convert_folder(
                paths["correct"],
                output_csv,
                patterns.get("movement_id", ""),
                patterns.get("correct_tag", "cor")
            )

            if isinstance(result, tuple):
                files_count, frames_count = result
                print(f"   ✅ Converted {files_count} files → {frames_count} frames")
            else:
                print(f"   ⚠️ No data converted")

    # ==========================================
    # 📊 SUMMARY
    # ==========================================
    print(f"\n{'=' * 55}")
    print("📊 CONVERSION SUMMARY:")
    for ex_key in exercises_to_convert:
        prefix = EXERCISES[ex_key]["csv_prefix"]
        for quality in ["Correct", "Incorrect", "Idle"]:
            fname = f"{prefix}_{quality}.csv"
            if os.path.exists(fname):
                # Count rows (subtract header)
                with open(fname, 'r') as f:
                    rows = sum(1 for _ in f) - 1
                print(f"   ✅ {fname}: {rows} frames")
            else:
                print(f"   ❌ {fname}: not found")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
