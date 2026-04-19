import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from exercise_config import (
    EXERCISES, EXPECTED_FEATURES, SEQUENCE_LENGTH,
    BRANCH_INDICES, ANGLE_FEATURE_INDICES
)

# ==========================================
# 🧠 POSEGURU: MULTI-EXERCISE MODEL TRAINER
# ==========================================
# Trains a SEPARATE model for each exercise.
# Each model classifies: 0=Correct, 1=Incorrect, 2=Idle
# Architecture: 4-Branch Hybrid Spatio-Temporal LSTM
# ==========================================


# ==========================================
# 🧩 BRANCH FEATURE EXTRACTORS
# ==========================================
def get_branch_features(x, landmark_indices):
    """Slices the array to grab specific 3D body part landmarks."""
    feature_indices = []
    for i in landmark_indices:
        feature_indices.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
    return tf.gather(x, feature_indices, axis=2)


def get_angle_features(x):
    """Slices the array to grab all 8 angle features (indices 132-139)."""
    return tf.gather(x, ANGLE_FEATURE_INDICES, axis=2)


# ==========================================
# 📂 DATA LOADING (PER EXERCISE)
# ==========================================
def load_exercise_data(exercise_key):
    """
    Load Correct, Incorrect, and Idle CSVs for a specific exercise.
    Returns (X_sequences, y_labels).
    """
    exercise = EXERCISES[exercise_key]
    prefix = exercise["csv_prefix"]

    correct_file = f"{prefix}_Correct.csv"
    incorrect_file = f"{prefix}_Incorrect.csv"
    idle_file = f"{prefix}_Idle.csv"

    print(f"\n📂 Loading data for {exercise['display_name']}...")

    sequences, labels = [], []

    # --- CORRECT DATA (label = 0) ---
    if os.path.exists(correct_file):
        df_correct = pd.read_csv(correct_file).values
        # Ensure columns match expected features
        if df_correct.shape[1] < EXPECTED_FEATURES:
            padding = np.zeros((df_correct.shape[0], EXPECTED_FEATURES - df_correct.shape[1]))
            df_correct = np.hstack([df_correct, padding])
        elif df_correct.shape[1] > EXPECTED_FEATURES:
            df_correct = df_correct[:, :EXPECTED_FEATURES]

        for i in range(0, len(df_correct) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            sequences.append(df_correct[i: i + SEQUENCE_LENGTH])
            labels.append(0)
        print(f"   ✅ {correct_file}: {len(df_correct)} frames → {labels.count(0)} sequences")
    else:
        print(f"   ❌ {correct_file}: NOT FOUND (skipping)")

    # --- INCORRECT DATA (label = 1) ---
    if os.path.exists(incorrect_file):
        df_incorrect = pd.read_csv(incorrect_file).values
        if df_incorrect.shape[1] < EXPECTED_FEATURES:
            padding = np.zeros((df_incorrect.shape[0], EXPECTED_FEATURES - df_incorrect.shape[1]))
            df_incorrect = np.hstack([df_incorrect, padding])
        elif df_incorrect.shape[1] > EXPECTED_FEATURES:
            df_incorrect = df_incorrect[:, :EXPECTED_FEATURES]

        count_before = len(labels)
        for i in range(0, len(df_incorrect) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            sequences.append(df_incorrect[i: i + SEQUENCE_LENGTH])
            labels.append(1)
        print(f"   ✅ {incorrect_file}: {len(df_incorrect)} frames → {len(labels) - count_before} sequences")
    else:
        print(f"   ❌ {incorrect_file}: NOT FOUND (skipping)")

    # --- IDLE DATA (label = 2) ---
    if os.path.exists(idle_file):
        df_idle = pd.read_csv(idle_file).values
        if df_idle.shape[1] < EXPECTED_FEATURES:
            padding = np.zeros((df_idle.shape[0], EXPECTED_FEATURES - df_idle.shape[1]))
            df_idle = np.hstack([df_idle, padding])
        elif df_idle.shape[1] > EXPECTED_FEATURES:
            df_idle = df_idle[:, :EXPECTED_FEATURES]

        count_before = len(labels)
        for i in range(0, len(df_idle) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            sequences.append(df_idle[i: i + SEQUENCE_LENGTH])
            labels.append(2)
        print(f"   ✅ {idle_file}: {len(df_idle)} frames → {len(labels) - count_before} sequences")
    else:
        print(f"   ❌ {idle_file}: NOT FOUND (skipping)")

    if len(sequences) == 0:
        print(f"   ❌ No data found for {exercise['display_name']}! Record data first.")
        return None, None

    return np.array(sequences, dtype=np.float32), np.array(labels)


# ==========================================
# 🧠 MODEL ARCHITECTURE
# ==========================================
def build_spatio_temporal_model():
    """
    4-Branch Hybrid Spatio-Temporal Architecture.
    Branches: Torso, Arms, Legs, Angles
    Input: (SEQUENCE_LENGTH, EXPECTED_FEATURES) = (30, 248)
    Output: 3 classes (Correct, Incorrect, Idle)
    """
    inputs = Input(shape=(SEQUENCE_LENGTH, EXPECTED_FEATURES))

    idx_torso = BRANCH_INDICES["torso"]
    idx_arms = BRANCH_INDICES["arms"]
    idx_legs = BRANCH_INDICES["legs"]

    # Create the 4 physical sub-networks
    branch_torso = Lambda(lambda x: get_branch_features(x, idx_torso), name="Torso_Split")(inputs)
    branch_arms = Lambda(lambda x: get_branch_features(x, idx_arms), name="Arms_Split")(inputs)
    branch_legs = Lambda(lambda x: get_branch_features(x, idx_legs), name="Legs_Split")(inputs)
    branch_angles = Lambda(get_angle_features, name="Angles_Split")(inputs)

    # Temporal Pyramids (LSTMs for each branch)
    lstm_torso = LSTM(32, return_sequences=False, name="Torso_LSTM")(branch_torso)
    lstm_arms = LSTM(32, return_sequences=False, name="Arms_LSTM")(branch_arms)
    lstm_legs = LSTM(32, return_sequences=False, name="Legs_LSTM")(branch_legs)
    lstm_angles = LSTM(32, return_sequences=False, name="Angles_LSTM")(branch_angles)  # 32 units for 8 angles

    # Hierarchical Merge
    merged = concatenate([lstm_torso, lstm_arms, lstm_legs, lstm_angles], name="Full_Body_Merge")

    # The Decision Maker
    dense_1 = Dense(128, activation='relu')(merged)
    drop_1 = Dropout(0.3)(dense_1)
    dense_2 = Dense(64, activation='relu')(drop_1)

    outputs = Dense(3, activation='softmax', name="Final_Decision")(dense_2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ==========================================
# 🏋️ TRAIN ONE EXERCISE
# ==========================================
def train_single_exercise(exercise_key):
    """Train and save a model for one exercise."""
    exercise = EXERCISES[exercise_key]
    model_name = f"{exercise['csv_prefix'].lower()}_specialist.h5"

    print(f"\n{'=' * 55}")
    print(f"  TRAINING: {exercise['display_name']}")
    print(f"  Model will be saved as: {model_name}")
    print(f"{'=' * 55}")

    # Load data
    X, y = load_exercise_data(exercise_key)
    if X is None:
        print(f"⏩ Skipping {exercise['display_name']} - no data available.")
        return False

    # One-hot encode labels
    y_hot = to_categorical(y, num_classes=3)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_hot, test_size=0.2, random_state=42
    )

    print(f"\n📊 Data Summary:")
    print(f"   Total sequences: {len(X)}")
    print(f"   Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"   Feature shape: {X.shape}")
    print(f"   Classes: Correct={list(y).count(0)}, Incorrect={list(y).count(1)}, Idle={list(y).count(2)}")

    # Build model
    print(f"\n🧠 Building 4-Branch Hybrid Spatio-Temporal Architecture...")
    model = build_spatio_temporal_model()
    model.summary()

    # Train
    print(f"\n🏃 Training {exercise['display_name']} Model...")
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Final Test Accuracy: {accuracy * 100:.1f}%")
    print(f"📉 Final Test Loss: {loss:.4f}")

    # Save
    model.save(model_name)
    print(f"\n🎉 Model saved: {model_name}")

    return True


# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
def main():
    print("=" * 55)
    print("   POSEGURU: MULTI-EXERCISE MODEL TRAINER")
    print("=" * 55)
    print("\nTrains a SEPARATE specialist model for each exercise.")
    print("Each model classifies: Correct | Incorrect | Idle\n")

    # ==========================================
    # 🎯 SELECT WHAT TO TRAIN
    # ==========================================
    exercise_keys = list(EXERCISES.keys())
    print("📋 What would you like to train?")
    for i, key in enumerate(exercise_keys):
        ex = EXERCISES[key]
        prefix = ex["csv_prefix"]
        model_file = f"{prefix.lower()}_specialist.h5"
        exists = "✅ (model exists)" if os.path.exists(model_file) else "❌ (not yet trained)"
        print(f"  {i + 1} - {ex['display_name']}  {exists}")
    print(f"  {len(exercise_keys) + 1} - Train ALL exercises")

    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice <= len(exercise_keys):
                break
            print("Invalid number!")
        except ValueError:
            print("Please enter a number!")

    # ==========================================
    # 🏋️ TRAIN SELECTED EXERCISE(S)
    # ==========================================
    if choice == len(exercise_keys):
        # Train all
        results = {}
        for key in exercise_keys:
            success = train_single_exercise(key)
            results[key] = success

        # Final summary
        print(f"\n{'=' * 55}")
        print("📊 TRAINING SUMMARY:")
        for key, success in results.items():
            ex = EXERCISES[key]
            model_file = f"{ex['csv_prefix'].lower()}_specialist.h5"
            if success:
                print(f"   ✅ {ex['display_name']}: {model_file}")
            else:
                print(f"   ❌ {ex['display_name']}: FAILED (no data)")
        print(f"{'=' * 55}")
    else:
        # Train single
        exercise_key = exercise_keys[choice]
        train_single_exercise(exercise_key)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
