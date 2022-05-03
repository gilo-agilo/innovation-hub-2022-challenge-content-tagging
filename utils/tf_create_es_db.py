import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

import keras
from keras.models import load_model

if __name__ == "__main__":
    model_path = "../saved-model/EfficientNetB3-car color-81.61.h5"
    model = load_model(model_path)
    # print(model.summary())

    # read images into one array
    image_path = "../static/ford/train"
    image_filenames = os.listdir(image_path)
    images = []
    for filename in tqdm(image_filenames, desc="image reading"):
        img_full = plt.imread(os.path.join(image_path, filename))
        img = cv2.resize(img_full, (224, 224))
        if img.shape != (224, 224, 3):
            print(img.shape, filename)
        images.append(img)
    images = np.array(images)
    print("images shape", images.shape)

    # get batch size for prediction
    steps = 50
    batch_size = len(image_filenames) // steps
    if len(image_filenames) % steps != 0.:
        batch_size += 1
    print("batch size =", batch_size)

    # get needed layer of model and predict features
    feature_extraction_model = keras.Model(inputs=model.input, outputs=model.layers[-4].output)

    # predict over batches of images
    features = None
    for batch in tqdm(range(batch_size), desc="batch prediction"):
        start_ind = batch * steps
        end_ind = min(batch * steps + steps, len(image_filenames))
        current_images = images[start_ind:end_ind]
        extracted_features = feature_extraction_model(current_images).numpy()
        if features is not None:
            features = np.concatenate((features, extracted_features), axis=0)
        else:
            features = extracted_features
    print("features shape", features.shape)

    # collect all features as list of jsons
    data = []
    for ind, filename in tqdm(enumerate(image_filenames), desc="creating ES DB"):
        filepath = os.path.join(image_path, filename)
        features_vec = features[ind]

        doc = {
            'id': filename,
            'filename': filename,
            'path': filepath,
            'features': features_vec.tolist()
        }
        data.append(doc)

    save_path = "../es_db/train_color.json"
    # save into json to file to be readable
    with open(save_path, "w") as outfile:
        json.dump(data, outfile)
    print(f"saved into {save_path}")

    pass
