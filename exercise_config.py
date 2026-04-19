# ==========================================
# ⚙️ POSEGURU: MULTI-EXERCISE CONFIGURATION
# ==========================================
# Central source of truth for all exercises.
# Every other script imports from this file.
# ==========================================

import numpy as np

# ==========================================
# 📐 GLOBAL CONSTANTS
# ==========================================
EXPECTED_FEATURES = 248        # 33 landmarks * 4 + 8 angles + padding
SEQUENCE_LENGTH = 30           # Frames per LSTM sequence
NUM_LANDMARKS = 33             # MediaPipe Pose landmarks
RAW_LANDMARK_FEATURES = NUM_LANDMARKS * 4   # 132 features (x, y, z, visibility)

# ==========================================
# 🦴 MEDIAPIPE LANDMARK INDICES
# ==========================================
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = {
    "nose": 0,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# ==========================================
# 📐 ANGLE DEFINITIONS (8 angles total)
# ==========================================
# Each angle is defined by 3 landmark indices: (A, B, C)
# The angle is measured at joint B, between segments BA and BC.
# These are computed in order and stored at feature indices 132–139.
ANGLE_DEFINITIONS = {
    "avg_knee":  {
        "left": (LM["left_hip"], LM["left_knee"], LM["left_ankle"]),      # 23, 25, 27
        "right": (LM["right_hip"], LM["right_knee"], LM["right_ankle"]),   # 24, 26, 28
        "index": 132,
        "description": "Average knee flexion angle"
    },
    "avg_hip": {
        "left": (LM["left_shoulder"], LM["left_hip"], LM["left_knee"]),    # 11, 23, 25
        "right": (LM["right_shoulder"], LM["right_hip"], LM["right_knee"]),# 12, 24, 26
        "index": 133,
        "description": "Average hip flexion angle"
    },
    "avg_ankle": {
        "left": (LM["left_knee"], LM["left_ankle"], LM["left_foot_index"]),   # 25, 27, 31
        "right": (LM["right_knee"], LM["right_ankle"], LM["right_foot_index"]),# 26, 28, 32
        "index": 134,
        "description": "Average ankle dorsiflexion angle"
    },
    "avg_spine": {
        "left": (LM["left_ear"], LM["left_shoulder"], LM["left_hip"]),     # 7, 11, 23
        "right": (LM["right_ear"], LM["right_shoulder"], LM["right_hip"]), # 8, 12, 24
        "index": 135,
        "description": "Average spine/torso alignment angle"
    },
    "left_knee": {
        "points": (LM["left_hip"], LM["left_knee"], LM["left_ankle"]),     # 23, 25, 27
        "index": 136,
        "description": "Left knee angle (for asymmetric exercises like lunges)"
    },
    "right_knee": {
        "points": (LM["right_hip"], LM["right_knee"], LM["right_ankle"]),  # 24, 26, 28
        "index": 137,
        "description": "Right knee angle (for asymmetric exercises like lunges)"
    },
    "left_hip": {
        "points": (LM["left_shoulder"], LM["left_hip"], LM["left_knee"]),  # 11, 23, 25
        "index": 138,
        "description": "Left hip angle"
    },
    "right_hip": {
        "points": (LM["right_shoulder"], LM["right_hip"], LM["right_knee"]),# 12, 24, 26
        "index": 139,
        "description": "Right hip angle"
    },
}

# Indices of all 8 angle features for the LSTM angle branch
ANGLE_FEATURE_INDICES = [132, 133, 134, 135, 136, 137, 138, 139]

# ==========================================
# 🏋️ SPATIO-TEMPORAL MODEL BODY PART GROUPS
# ==========================================
# MediaPipe landmark indices for each body region branch
BRANCH_INDICES = {
    "torso": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24],
    "arms":  [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "legs":  [25, 26, 27, 28, 29, 30, 31, 32],
}

# ==========================================
# 🏋️ EXERCISE DEFINITIONS
# ==========================================
EXERCISES = {
    "squat": {
        "name": "Squat",
        "display_name": "🏋️ Squat",
        "description": "A compound lower-body exercise targeting quads, glutes, and hamstrings.",

        # Which angles are CRITICAL to monitor for this exercise
        "primary_angles": ["avg_knee", "avg_hip"],
        "secondary_angles": ["avg_spine"],

        # Angles shown on the live UI
        "display_angles": {
            "Knee": "avg_knee",
            "Hip": "avg_hip",
        },

        # Correct form ranges (in degrees)
        "correct_ranges": {
            "avg_knee": {"min": 70, "max": 110, "label": "Knee Flexion"},
            "avg_hip":  {"min": 70, "max": 120, "label": "Hip Hinge"},
        },

        # 🛑 FEEDBACK RULES (checked in order, first match wins)
        # Each rule: (angle_name, condition, threshold, message, severity)
        # condition: "below" = angle < threshold, "above" = angle > threshold
        "feedback_rules": [
            ("avg_hip",  "below", 75,  "STRAIGHTEN YOUR BACK",  "danger"),
            ("avg_knee", "above", None, "GO DOWN MORE",          "warning"),  # None = uses calibrated target + offset
            ("avg_knee", "below", 60,  "TOO DEEP - COME UP",    "danger"),
        ],
        "perfect_message": "PERFECT SQUAT!",

        # 🔄 REP DETECTION THRESHOLDS
        "rep_detection": {
            "up_knee": 160,       # Knee angle > this = standing up
            "up_hip": 155,        # Hip angle > this = standing up
            "down_knee_offset": 15,  # How far below target to count as "down"
            "angle_for_down": "avg_knee",
            "angle_for_up_knee": "avg_knee",
            "angle_for_up_hip": "avg_hip",
        },

        # 🎯 CALIBRATION
        "calibration": {
            "angle": "avg_knee",
            "instruction": "Do your DEEPEST safe squat and hold for 2 seconds",
            "default_target": 120,
            "offset": 10,   # Add to lowest recorded angle
        },

        # Visibility check landmarks
        "required_landmarks": [23, 24, 25, 26, 27, 28],

        # CSV file naming
        "csv_prefix": "Squat",
    },

    "lunge": {
        "name": "Lunge",
        "display_name": "🦵 Lunge",
        "description": "A unilateral exercise targeting quads, glutes, and balance.",

        "primary_angles": ["left_knee", "right_knee", "avg_hip"],
        "secondary_angles": ["avg_spine", "avg_ankle"],

        "display_angles": {
            "Front Knee": "left_knee",      # Leading leg (user can switch)
            "Back Knee": "right_knee",       # Trailing leg
            "Torso": "avg_hip",
        },

        "correct_ranges": {
            "left_knee":  {"min": 80, "max": 100, "label": "Front Knee"},
            "right_knee": {"min": 80, "max": 110, "label": "Back Knee"},
            "avg_hip":    {"min": 150, "max": 180, "label": "Torso Upright"},
        },

        "feedback_rules": [
            ("avg_hip",    "below", 145, "KEEP TORSO UPRIGHT",     "danger"),
            ("left_knee",  "below", 70,  "FRONT KNEE TOO DEEP",   "danger"),
            ("left_knee",  "above", None, "LUNGE DEEPER",          "warning"),
            ("right_knee", "above", 130, "DROP BACK KNEE LOWER",  "warning"),
        ],
        "perfect_message": "PERFECT LUNGE!",

        "rep_detection": {
            "up_knee": 155,
            "up_hip": 160,
            "down_knee_offset": 15,
            "angle_for_down": "left_knee",
            "angle_for_up_knee": "avg_knee",
            "angle_for_up_hip": "avg_hip",
        },

        "calibration": {
            "angle": "left_knee",
            "instruction": "Do your DEEPEST safe lunge and hold for 2 seconds",
            "default_target": 95,
            "offset": 5,
        },

        "required_landmarks": [23, 24, 25, 26, 27, 28],
        "csv_prefix": "Lunge",
    },

    "glute_bridge": {
        "name": "Glute Bridge",
        "display_name": "🍑 Glute Bridge",
        "description": "A hip extension exercise targeting glutes and hamstrings. Performed lying down.",

        "primary_angles": ["avg_hip", "avg_knee"],
        "secondary_angles": ["avg_spine"],

        "display_angles": {
            "Hip Extension": "avg_hip",
            "Knee": "avg_knee",
            "Spine": "avg_spine",
        },

        "correct_ranges": {
            "avg_hip":   {"min": 165, "max": 185, "label": "Hip Extension at Top"},
            "avg_knee":  {"min": 75, "max": 105, "label": "Knee Bend"},
            "avg_spine": {"min": 155, "max": 185, "label": "Spine Neutral"},
        },

        "feedback_rules": [
            ("avg_hip",   "below", None, "PUSH HIPS HIGHER",        "warning"),
            ("avg_spine", "below", 145,  "KEEP SPINE NEUTRAL",      "danger"),
            ("avg_hip",   "above", 190,  "DON'T OVEREXTEND",        "danger"),
        ],
        "perfect_message": "PERFECT BRIDGE!",

        "rep_detection": {
            "up_knee": None,      # Not used for bridges
            "up_hip": None,       # Not used for bridges
            "down_hip": 130,      # Hip angle < this = hips are down (resting)
            "top_hip": 165,       # Hip angle > this = top of bridge
            "angle_for_down": "avg_hip",
            "angle_for_up_knee": None,
            "angle_for_up_hip": "avg_hip",
        },

        "calibration": {
            "angle": "avg_hip",
            "instruction": "Push your hips UP as high as you safely can and hold for 2 seconds",
            "default_target": 170,
            "offset": -5,  # Subtract from highest (we want them to reach close to max)
        },

        "required_landmarks": [11, 12, 23, 24, 25, 26],
        "csv_prefix": "GluteBridge",
    },
}

# ==========================================
# 🧮 UTILITY: ANGLE CALCULATOR
# ==========================================
def calculate_angle(a, b, c):
    """
    Calculate the angle at point B given three landmark objects (with .x, .y attributes).
    Returns angle in degrees (0-360 mapped to 0-180).
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def calculate_angle_from_coords(a, b, c):
    """
    Calculate angle from raw (x,y) coordinate tuples/arrays.
    Used when working with arrays instead of landmark objects.
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def compute_all_angles(landmarks):
    """
    Given MediaPipe pose landmarks, compute all 8 angles.
    Returns a dict: {"avg_knee": value, "avg_hip": value, ...}
    """
    lm = landmarks
    angles = {}

    # Averaged angles (left + right / 2)
    for name in ["avg_knee", "avg_hip", "avg_ankle", "avg_spine"]:
        defn = ANGLE_DEFINITIONS[name]
        left_angle = calculate_angle(
            lm[defn["left"][0]], lm[defn["left"][1]], lm[defn["left"][2]]
        )
        right_angle = calculate_angle(
            lm[defn["right"][0]], lm[defn["right"][1]], lm[defn["right"][2]]
        )
        angles[name] = (left_angle + right_angle) / 2.0

    # Individual side angles
    for name in ["left_knee", "right_knee", "left_hip", "right_hip"]:
        defn = ANGLE_DEFINITIONS[name]
        pts = defn["points"]
        angles[name] = calculate_angle(lm[pts[0]], lm[pts[1]], lm[pts[2]])

    return angles


def angles_to_feature_list(angles_dict):
    """
    Convert the angles dictionary to an ordered list matching feature indices 132-139.
    """
    ordered_names = ["avg_knee", "avg_hip", "avg_ankle", "avg_spine",
                     "left_knee", "right_knee", "left_hip", "right_hip"]
    return [angles_dict[name] for name in ordered_names]


def build_feature_vector(landmarks, angles_dict):
    """
    Build the full 248-feature vector from landmarks and computed angles.
    """
    features = []

    # 1. Raw landmark coordinates (132 features: 33 * 4)
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    # 2. Computed angles (8 features: indices 132-139)
    features.extend(angles_to_feature_list(angles_dict))

    # 3. Zero-pad to EXPECTED_FEATURES
    features.extend([0.0] * (EXPECTED_FEATURES - len(features)))

    return features


def get_exercise_list():
    """Return list of exercise keys."""
    return list(EXERCISES.keys())


def get_exercise_name(key):
    """Return display name for an exercise key."""
    return EXERCISES[key]["display_name"]
