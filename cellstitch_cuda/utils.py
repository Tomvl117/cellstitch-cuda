import os


def get_filenames(data_path):
    filenames = []

    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            filenames.append(file)
    return filenames
