import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from predict_model import verify_person


DATASET_PATH = "../dataset"


def evaluate():

    y_true = []
    y_pred = []

    persons = os.listdir(DATASET_PATH)

    for person in persons:

        person_path = os.path.join(
            DATASET_PATH,
            person
        )

        if not os.path.isdir(person_path):
            continue

        test_folder = os.path.join(
            person_path,
            "test"
        )

        for video in os.listdir(test_folder):

            if video.startswith("."):
                continue

            video_path = os.path.join(
                test_folder,
                video
            )

            for target_person in persons:

                target_path = os.path.join(
                    DATASET_PATH,
                    target_person
                )

                if not os.path.isdir(target_path):
                    continue

                status, confidence = verify_person(
                    target_person,
                    video_path
                )

                if person == target_person:
                    y_true.append(1)
                else:
                    y_true.append(0)

                if status == "VERIFIED":
                    y_pred.append(1)
                else:
                    y_pred.append(0)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title("Gait Verification Confusion Matrix")

    plt.show()


if __name__ == "__main__":

    evaluate()