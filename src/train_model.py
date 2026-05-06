import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_features
import random

DATASET_PATH = "/Users/aditi/Desktop/gait_project/dataset"
MODEL_PATH = "/Users/aditi/Desktop/gait_project/models"

# How many negative samples per positive
NEGATIVE_MULTIPLIER = 1


def train_person_model(person_name):

    print(f"\nTraining model for: {person_name}")

    positive_features = []
    negative_features = []

    persons = [
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, p)
        )
    ]

    # -----------------------------
    # Collect positive samples
    # -----------------------------

    pos_folder = os.path.join(
        DATASET_PATH,
        person_name,
        "train"
    )

    if not os.path.exists(pos_folder):

        print("Skipping — no train folder")

        return

    videos = [
        v for v in os.listdir(pos_folder)
        if not v.startswith(".")
    ]

    if len(videos) == 0:

        print("Skipping — empty train folder")

        return

    for video in videos:

        video_path = os.path.join(
            pos_folder,
            video
        )

        features = extract_features(
            video_path
        )

        if features is not None:

            positive_features.append(
                features
            )

    if len(positive_features) == 0:

        print("No valid positive features")

        return

    print(
        "Positive samples:",
        len(positive_features)
    )

    # -----------------------------
    # Collect negative samples
    # -----------------------------

    for other_person in persons:

        if other_person == person_name:

            continue

        neg_folder = os.path.join(
            DATASET_PATH,
            other_person,
            "train"
        )

        if not os.path.exists(neg_folder):

            continue

        for video in os.listdir(neg_folder):

            if video.startswith("."):

                continue

            video_path = os.path.join(
                neg_folder,
                video
            )

            features = extract_features(
                video_path
            )

            if features is not None:

                negative_features.append(
                    features
                )

    if len(negative_features) == 0:

        print("No negative samples found")

        return

    # -----------------------------
    # Balance negatives
    # -----------------------------

    max_negatives = (
        len(positive_features)
        * NEGATIVE_MULTIPLIER
    )

    if len(negative_features) > max_negatives:

        negative_features = random.sample(
            negative_features,
            max_negatives
        )

    print(
        "Negative samples:",
        len(negative_features)
    )

    # -----------------------------
    # Build dataset
    # -----------------------------

    X = (
        positive_features
        + negative_features
    )

    y = (
        [1] * len(positive_features)
        + [0] * len(negative_features)
    )

    X = np.array(X)
    y = np.array(y)

    print(
        "Total training samples:",
        len(X)
    )

    # -----------------------------
    # Feature scaling
    # -----------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # -----------------------------
    # Train model
    # -----------------------------

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced"

    )

    model.fit(X, y)

    # -----------------------------
    # Save model
    # -----------------------------

    os.makedirs(
        MODEL_PATH,
        exist_ok=True
    )

    model_file = os.path.join(
        MODEL_PATH,
        f"{person_name}_model.pkl"
    )

    scaler_file = os.path.join(
        MODEL_PATH,
        f"{person_name}_scaler.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    print(
        "Model saved successfully"
    )


def train_all():

    persons = [
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, p)
        )
    ]

    print("\nPersons detected:")

    for p in persons:

        print("-", p)

    for person in persons:

        train_person_model(person)


if __name__ == "__main__":

    train_all()
