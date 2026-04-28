import joblib
import numpy as np
from src.feature_extraction import extract_features
import os

MODEL_PATH = "/Users/aditi/Desktop/gait_project/models"

def verify_person(person_name, video_path):

    model_file = f"{MODEL_PATH}/{person_name}_model.pkl"
    scaler_file = f"{MODEL_PATH}/{person_name}_scaler.pkl"

    if not os.path.exists(model_file):

        print("Model not found")

        return "NOT VERIFIED", 0.0

    model = joblib.load(model_file)

    scaler = joblib.load(scaler_file)

    print("Extracting features...")

    features = extract_features(video_path)

    if features is None:

        print("No gait detected")

        return "NOT VERIFIED", 0.0

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    probability = model.predict_proba(features)[0][1]

    threshold = 0.65

    if probability >= threshold:

        status = "VERIFIED"

    else:

        status = "NOT VERIFIED"

    print()
    print("Prediction Result")
    print("Status:", status)
    print("Confidence:", round(probability, 2))

    return status, probability

if __name__ == "__main__":

    person = input("Enter person name: ")

    video = input("Enter test video path: ")

    verify_person(
        person,
        video
    )