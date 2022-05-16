import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    labels = pd.read_csv("../data/ford_files.csv")
    labels["FileName"] = list(map(lambda x: x+".jpg" if ".jpg" not in x.lower() else x, labels["FileName"].values))
    labels.drop_duplicates(["FileName"], inplace=True)

    with open("../output/test_car_model.pickle", "rb") as file:
        predictions = pickle.load(file)

    accuracy_mask = []
    for query in predictions:
        query_filepath = query["query_id"]
        found_images = query["images"]
        found_filenames = list(map(lambda x: x["filename"], found_images))
        found_scores = list(map(lambda x: x["score"], found_images))
        found_dict = dict(zip(found_filenames, found_scores))

        query_labels = labels[labels["FileName"] == query_filepath].reset_index(drop=True)
        if query_labels.shape[0] > 0:
            found_labels = labels[labels["FileName"].isin(found_filenames)].reset_index(drop=True)
            found_labels["Score"] = [found_dict[x] for x in found_labels["FileName"].values]

            mask = found_labels.values[:, 1:-1] == query_labels.values[:, 1:]
            accuracy_mask = np.vstack((accuracy_mask, mask)) if len(accuracy_mask) > 0 else mask

    accuracy_mask = accuracy_mask.astype(int)
    true_sum = np.sum(accuracy_mask, axis=0)
    accuracy = true_sum / accuracy_mask.shape[0]
    accuracy_df = pd.DataFrame(data=np.expand_dims(accuracy, axis=0), columns=labels.columns.values[1:])

    # accuracy per value
    columns_names, column_values, column_counts = [], [], []
    for column in labels.columns.values[1:]:
        unique_values = labels[column].value_counts()
        column_values.extend(unique_values.index.values.tolist())
        # FIXME
        column_counts.extend(unique_values.values.tolist())
        columns_names.extend([column] * len(unique_values))
    labels_values_df = pd.DataFrame()
    labels_values_df["column"] = columns_names
    labels_values_df["value"] = column_values
    labels_values_df["value count"] = column_counts
    labels_values_df["labels"] = [None] * len(column_values)
    labels_values_df["score"] = [None] * len(column_values)

    accuracy_mask = []
    for query in predictions:
        query_filepath = query["query_id"]
        found_images = query["images"]
        found_filenames = list(map(lambda x: x["filename"], found_images))
        found_scores = list(map(lambda x: x["score"], found_images))
        found_dict = dict(zip(found_filenames, found_scores))

        query_labels = labels[labels["FileName"] == query_filepath].reset_index(drop=True)
        if query_labels.shape[0] > 0:
            found_labels = labels[labels["FileName"].isin(found_filenames)].reset_index(drop=True)
            found_labels["Score"] = [found_dict[x] for x in found_labels["FileName"].values]

            mask = found_labels.values[:, 1:-1] == query_labels.values[:, 1:]
            for ind, column in enumerate(query_labels.columns.values[1:]):
                current_values = labels_values_df[(labels_values_df["column"] == column) &
                                                  (labels_values_df["value"] == query_labels[column][0])]["labels"].values[0]
                if current_values is not None:
                    current_values = np.hstack((current_values, mask[:, ind]))
                else:
                    current_values = mask[:, ind]
                index2insert = labels_values_df[(labels_values_df["column"] == column) &
                                                (labels_values_df["value"] == query_labels[column][0])].index.values[0]
                labels_values_df.at[index2insert, "labels"] = np.array(current_values).astype(object)

    for ind, row in labels_values_df.iterrows():
        if row["labels"] is not None:
            labels_values_df.at[ind, "score"] = round(sum(row["labels"]) / len(row["labels"]), 3)
    labels_values_df.drop(["labels"], axis=1, inplace=True)

    pass
