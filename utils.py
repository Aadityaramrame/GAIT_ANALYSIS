import os


def get_person_list():

    dataset_path = "/Users/aditi/Desktop/gait_project/dataset"

    persons = []

    for person in os.listdir(dataset_path):

        if os.path.isdir(
            os.path.join(dataset_path, person)
        ):
            persons.append(person)

    return persons