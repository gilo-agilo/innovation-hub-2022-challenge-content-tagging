"""
divide into train and test sets with balanced Model, Color, Background
move correspondent images to train and test folders
"""
import os.path
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
from tqdm import tqdm

from constants import *

if __name__ == "__main__":
    labels_path = "../data/car_models.csv"
    labels = pd.read_csv(labels_path, header=None)
    labels.columns = ["FileName", "Model"]

    columns = ["Model"]
    # drop column values which met just once
    value_counts = labels[columns].value_counts()
    values_1 = value_counts[value_counts == 1].index.values
    values_1 = np.array(list(map(lambda x: np.array(x).astype(str), values_1)))
    labels_subset = deepcopy(labels)
    for value_1 in values_1:
        mask = (labels_subset[columns] == value_1).values.all(axis=1)
        labels_subset = labels_subset[~mask]
    labels_subset = labels_subset.reset_index(drop=True)

    # get only subset with missed models
    missed_models = ["c-max", "explorer", "kuga"]
    missed_models_inds = list(map(lambda x: CAR_MODEL_LABELS[x], missed_models))
    labels_subset = labels_subset[labels_subset["Model"].isin(missed_models_inds)].reset_index(drop=True)

    X, y = labels_subset["FileName"], labels_subset[columns]

    # split into train and test subsets
    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stratSplit.get_n_splits(X, y)
    for train_index, test_index in stratSplit.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]

        print("train = \n", y_train.value_counts())
        print("test = \n", y_test.value_counts())

    # concatenate labels and values
    train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    # save test filenames separately to move them in S3
    test.to_csv("../data/test_move.csv", index=False, header=False)

    # concatenate with existing dataset
    test_initial = pd.read_csv("../data/car_models_test_bucket.csv", header=None)
    test_initial.columns = ["FileName", "Model"]

    test_all = pd.concat([test_initial, test], axis=0).reset_index(drop=True)
    test_all.to_csv("../data/car_models_test_bucket_full.csv", index=False, header=False)

    # train.to_csv("../data/train.csv", index=False)
    # test.to_csv("../data/test.csv", index=False)
    #
    # # move images
    # images_path = r"C:\Users\ann\Code\challenges\Initial Images\Ford Images"
    # train_path = r"../static/ford/train"
    # test_path = r"../static/ford/test"
    #
    # # for train
    # for filename in tqdm(train["FileName"].values, desc="moving train images"):
    #     if ".jpg" not in filename:
    #         filename += ".jpg"
    #     source_path = os.path.join(images_path, filename)
    #     if not os.path.exists(source_path):
    #         print(source_path)
    #         continue
    #
    #     destination_path = os.path.join(train_path, filename)
    #     shutil.move(source_path, destination_path)
    #
    # print()
    #
    # # for test
    # for filename in tqdm(test["FileName"].values, desc="moving test images"):
    #     if ".jpg" not in filename:
    #         filename += ".jpg"
    #     source_path = os.path.join(images_path, filename)
    #     if not os.path.exists(source_path):
    #         print(source_path)
    #         continue
    #
    #     destination_path = os.path.join(test_path, filename)
    #     shutil.move(source_path, destination_path)

    pass
