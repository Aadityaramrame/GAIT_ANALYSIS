import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------

DATA_PATH = "/Users/aditi/Desktop/gait_project/dataset/person1"

TRAIN_FILES = [
    "/Users/aditi/Desktop/gait_project/dataset/person1/train1.MOV",
    "/Users/aditi/Desktop/gait_project/dataset/person1/train2.MOV",
    "/Users/aditi/Desktop/gait_project/dataset/person1/train3.MOV"
]

TEST_FILE = "/Users/aditi/Desktop/gait_project/dataset/person1/test.MOV"

FRAME_SAMPLE_RATE = 5
MIN_FRAMES_REQUIRED = 10

MODEL_FILE = "gait_model.pkl"
SCALER_FILE = "scaler.pkl"

# -----------------------------
# Initialize MediaPipe
# -----------------------------

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Gait Cycle Detection
# -----------------------------

def detect_gait_cycles(ankle_positions):

    peaks = []

    for i in range(1, len(ankle_positions) - 1):

        if (
            ankle_positions[i] >
            ankle_positions[i - 1]
            and
            ankle_positions[i] >
            ankle_positions[i + 1]
        ):
            peaks.append(i)

    return len(peaks)

# -----------------------------
# Feature Engineering
# -----------------------------

def compute_features(landmarks):

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    left_knee = landmarks[25]
    right_knee = landmarks[26]

    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    stride_length = abs(
        left_ankle.x - right_ankle.x
    )

    step_width = abs(
        left_ankle.y - right_ankle.y
    )

    hip_width = abs(
        left_hip.x - right_hip.x
    )

    knee_height = abs(
        left_knee.y - left_ankle.y
    )

    symmetry = abs(
        (left_knee.y - left_ankle.y)
        -
        (right_knee.y - right_ankle.y)
    )

    return [
        stride_length,
        step_width,
        hip_width,
        knee_height,
        symmetry
    ]

# -----------------------------
# Feature Extraction
# -----------------------------

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    all_features = []

    ankle_positions = []

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Frame Sampling

        if frame_count % FRAME_SAMPLE_RATE != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        if results.pose_landmarks:

            landmarks = (
                results
                .pose_landmarks
                .landmark
            )

            # Save ankle motion for gait cycle

            ankle_positions.append(
                landmarks[27].y
            )

            features = compute_features(
                landmarks
            )

            all_features.append(
                features
            )

    cap.release()

    # Quality Check

    if len(all_features) < MIN_FRAMES_REQUIRED:

        print(
            "Low quality video:",
            video_path
        )

        return None

    # Gait cycle detection

    gait_cycles = detect_gait_cycles(
        ankle_positions
    )

    mean_features = np.mean(
        all_features,
        axis=0
    )

    final_features = list(
        mean_features
    )

    final_features.append(
        gait_cycles
    )

    return final_features

# -----------------------------
# Load Dataset
# -----------------------------

def load_dataset():

    X = []
    y = []

    print("\nExtracting features...\n")

    for file in TRAIN_FILES:

        path = os.path.join(
            DATA_PATH,
            file
        )

        print("Processing:", file)

        features = extract_features(
            path
        )

        if features is not None:

            X.append(features)

            y.append(1)

    # Create negative samples (required)

    for file in TRAIN_FILES:

        path = os.path.join(
            DATA_PATH,
            file
        )

        features = extract_features(
            path
        )

        if features is not None:

            noise = np.random.normal(
                0,
                0.01,
                size=len(features)
            )

            X.append(
                np.array(features) + noise
            )

            y.append(0)

    return (
        np.array(X),
        np.array(y)
    )

# -----------------------------
# Training Pipeline
# -----------------------------

def train_model():

    X, y = load_dataset()

    print(
        "\nDataset size:",
        len(X)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=100
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "\nAccuracy:",
        round(accuracy, 3)
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print(
        "Model saved"
    )

# -----------------------------
# Prediction Pipeline
# -----------------------------

def predict():

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    test_path = os.path.join(
        DATA_PATH,
        TEST_FILE
    )

    print(
        "\nTesting video...\n"
    )

    features = extract_features(
        test_path
    )

    features = scaler.transform(
        [features]
    )

    prediction = model.predict(
        features
    )[0]

    confidence = model.predict_proba(
        features
    )[0].max()

    if prediction == 1:

        print(
            "MATCH — Person verified"
        )

    else:

        print(
            "NOT MATCH — Unknown person"
        )

    print(
        "Confidence:",
        round(confidence, 3)
    )

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    print(
        "\n--- TRAINING MODEL ---"
    )

    train_model()

    print(
        "\n--- RUNNING PREDICTION ---"
    )

    predict()
