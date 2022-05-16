import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from constants import *

if __name__ == "__main__":
    images_path = r"C:\Users\ann\Code\challenges\datasets\crawler_outside"

    models_list = os.listdir(images_path)
    colors_dict = {}
    for car_model in models_list:
        model_path = os.path.join(images_path, car_model)
        colors_dict[car_model] = os.listdir(model_path)

    # save whole dataset
    # cars_path, cars_label = [], []
    # for car_model, colors_list in colors_dict.items():
    #     for car_color in colors_list:
    #         current_path = f"{car_model}/{car_color}"
    #         filenames = os.listdir(os.path.join(images_path, current_path))
    #         for filename in filenames:
    #             cars_path.append(f"{current_path}/{filename}")
    #             cars_label.append(CAR_MODEL_LABELS[car_model])
    #
    # car_models_labels = pd.DataFrame()
    # car_models_labels["model"] = cars_path
    # car_models_labels["label"] = cars_label
    # car_models_labels.to_csv("../data/car_models.csv", index=False, header=False)

    # prepare file for test
    labels_path = "../data/ford_files.csv"
    labels = pd.read_csv(labels_path)
    # labels["filename_ext"] = [x+".jpg" if ".jpg" not in x.lower() else x for x in labels["FileName"].values]

    test_images_path = r"C:\Users\ann\Code\challenges\datasets\ford"
    test_images_foldes = os.listdir(test_images_path)
    cars_path, cars_label = [], []
    for folder in test_images_foldes:
        filenames = os.listdir(os.path.join(test_images_path, folder))
        for filename in filenames:
            search_name = Path(filename).stem
            if search_name in labels["FileName"].values:
                car_model = labels[labels["FileName"] == search_name]["Model"].values[0]
                cars_label.append(CAR_MODEL_LABELS[car_model])
                cars_path.append(f"{folder}/{filename}")

    car_models_labels = pd.DataFrame()
    car_models_labels["model"] = cars_path
    car_models_labels["label"] = cars_label
    car_models_labels.to_csv("../data/car_models_test.csv", index=False, header=False)

    pass
