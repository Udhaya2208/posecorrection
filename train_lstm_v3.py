import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 30
EXPECTED_FEATURES = 240
MODEL_NAME = "squat_specialist_v3.h5"

# ==========================================
# 📂 2. DATA LOADING
# ==========================================
def load_data():
    print("⏳ Loading CSV Data...")
    df_correct = pd.read_csv("Data_Correct.csv").values
    df_incorrect = pd.read_csv("Data_Incorrect.csv").values
    df_idle = pd.read_csv("Data_Idle.csv").values

    sequences, labels = [], []

    # 0 = Good, 1 = Bad, 2 = Idle
    for i in range(0, len(df_correct) - SEQUENCE_LENGTH, SEQUENCE_LENGTH): 
        sequences.append(df_correct[i : i + SEQUENCE_LENGTH]); labels.append(0) 
    for i in range(0, len(df_incorrect) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
        sequences.append(df_incorrect[i : i + SEQUENCE_LENGTH]); labels.append(1) 
    for i in range(0, len(df_idle) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
        sequences.append(df_idle[i : i + SEQUENCE_LENGTH]); labels.append(2) 

    return np.array(sequences), np.array(labels)

# ==========================================
# 🧠 3. HYBRID SPATIO-TEMPORAL ARCHITECTURE
# ==========================================
def get_branch_features(x, landmark_indices):
    """Slices the array to grab specific 3D body parts."""
    feature_indices = []
    for i in landmark_indices:
        feature_indices.extend([i*4, i*4+1, i*4+2, i*4+3])
    return tf.gather(x, feature_indices, axis=2)

def get_angle_features(x):
    """Slices the array to grab strictly the Knee and Hip angles (Indices 132, 133)"""
    return tf.gather(x, [132, 133], axis=2)

def build_spatio_temporal_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, EXPECTED_FEATURES))

    # MediaPipe Landmark Indices
    idx_torso = [0,1,2,3,4,5,6,7,8,9,10, 11,12, 23,24]
    idx_arms = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    idx_legs = [25, 26, 27, 28, 29, 30, 31, 32]

    # Create the 4 physical sub-networks
    branch_torso  = Lambda(lambda x: get_branch_features(x, idx_torso), name="Torso_Split")(inputs)
    branch_arms   = Lambda(lambda x: get_branch_features(x, idx_arms), name="Arms_Split")(inputs)
    branch_legs   = Lambda(lambda x: get_branch_features(x, idx_legs), name="Legs_Split")(inputs)
    branch_angles = Lambda(get_angle_features, name="Angles_Split")(inputs) # 🚀 NEW ANGLE BRANCH!

    # Temporal Pyramids (LSTMs for each branch)
    lstm_torso  = LSTM(32, return_sequences=False, name="Torso_LSTM")(branch_torso)
    lstm_arms   = LSTM(32, return_sequences=False, name="Arms_LSTM")(branch_arms)
    lstm_legs   = LSTM(32, return_sequences=False, name="Legs_LSTM")(branch_legs)
    lstm_angles = LSTM(16, return_sequences=False, name="Angles_LSTM")(branch_angles) # 🚀 NEW ANGLE MEMORY!

    # Hierarchical Merge (Combine all 4 thoughts)
    merged = concatenate([lstm_torso, lstm_arms, lstm_legs, lstm_angles], name="Full_Body_Merge")

    # The Decision Maker
    dense_1 = Dense(128, activation='relu')(merged)
    drop_1  = Dropout(0.3)(dense_1)
    dense_2 = Dense(64, activation='relu')(drop_1)
    
    outputs = Dense(3, activation='softmax', name="Final_Decision")(dense_2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 🚀 4. EXECUTION
# ==========================================
def main():
    X, y = load_data()
    y_hot = to_categorical(y, num_classes=3) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.2, random_state=42)
    
    print("\n🧠 Building 4-Branch Hybrid Spatio-Temporal Architecture...")
    model = build_spatio_temporal_model()
    model.summary() 
    
    print("\n🏃 Training Advanced Model...")
    model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))
    
    model.save(MODEL_NAME)
    print(f"\n🎉 MASTERPIECE SAVED! Your new angle-aware {MODEL_NAME} is ready.")

if __name__ == "__main__":
    main()