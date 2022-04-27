"""
parse QA labels to make uniform
"""
import pandas as pd

if __name__ == "__main__":
    labels_path = "../data/Ford files.xlsx"
    labels = pd.read_excel(labels_path)
    for column in labels.columns.values[1:]:
        labels[column] = labels[column].apply(lambda x: x.lower())

    labels["Color"] = labels["Color"].replace("dark grey", "grey")
    labels["Color"] = labels["Color"].replace("dark blue", "blue")
    labels["Color"] = labels["Color"].replace("light grey", "grey")

    labels["Background"] = labels["Background"].replace("mountain", "mountains")
    labels["Background"] = labels["Background"].replace("moutains", "mountains")
    labels["Background"] = labels["Background"].replace("grey backround", "plain")
    labels["Background"] = labels["Background"].replace("grey background", "plain")
    labels["Background"] = labels["Background"].replace("blue backround", "plain")
    labels["Background"] = labels["Background"].replace("white background", "plain")
    labels["Background"] = labels["Background"].replace("white backround", "plain")
    labels["Background"] = labels["Background"].replace("blue background", "plain")
    labels["Background"] = labels["Background"].replace("dark background", "plain")
    labels["Background"] = labels["Background"].replace("nature/sea", "sea")
    labels["Background"] = labels["Background"].replace("sea/nature", "sea")
    labels["Background"] = labels["Background"].replace("street/ sea", "sea")
    labels["Background"] = labels["Background"].replace("nature/road", "street")
    labels["Background"] = labels["Background"].replace("nature/snow", "nature")

    for column in labels.columns.values[1:]:
        print('\n', column)
        print(labels[column].value_counts())

    labels.to_csv("../data/ford_files.csv", index=False)

    images_path = r"C:\Users\ann\Code\challenges\Ford Images"

    pass
