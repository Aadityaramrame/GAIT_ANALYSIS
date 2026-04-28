import cv2
import numpy as np
import mediapipe as mp
import os


mp_pose = mp.solutions.pose

FEATURE_PATH = "/Users/aditi/Desktop/gait_project/features"


def extract_features(video_path):

    person_name = video_path.split("/")[-3]

    video_name = os.path.basename(video_path)

    feature_dir = os.path.join(
        FEATURE_PATH,
        person_name
    )

    os.makedirs(feature_dir, exist_ok=True)

    feature_file = os.path.join(
        feature_dir,
        video_name.replace(".mp4", ".npy")
        .replace(".MOV", ".npy")
        .replace(".mov", ".npy")
    )

    # -----------------------------
    # Load cached feature if exists
    # -----------------------------

    if os.path.exists(feature_file):

        return np.load(feature_file)

    # -----------------------------
    # Otherwise compute features
    # -----------------------------

    cap = cv2.VideoCapture(video_path)

    pose = mp_pose.Pose()

    features = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = pose.process(rgb)

        if result.pose_landmarks:

            landmarks = result.pose_landmarks.landmark

            frame_features = []

            for lm in landmarks:

                frame_features.append(lm.x)
                frame_features.append(lm.y)

            features.append(frame_features)

    cap.release()

    if len(features) == 0:

        return None

    features = np.array(features)

    features = np.mean(
        features,
        axis=0
    )

    # -----------------------------
    # Save cached feature
    # -----------------------------

    np.save(
        feature_file,
        features
    )

    return features