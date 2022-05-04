import os
import sys
import cv2
from flask import Flask
import logging
import json
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import keras
from keras.models import load_model
from elasticsearch import Elasticsearch

from index.searcher import Searcher
from index.indexer import Indexer

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

ES_PASSWORD = "+b77YVyI_QDtEAMO=bRl"
ES_DB_PATH = "../es_db/train_color.json"

hosts = "http://localhost:9200"
index_name = 'cifar10'
number_of_shards = 30
number_of_replicas = 0


def tf_features_vector(model, image):
    if model is None:
        app.logger.error(f"Provided deep-learning model is None ...")
        return None

    # prepare image to appropriate shape
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    if img.shape != (224, 224, 3):
        app.logger.error("query image shape is not (224, 224, 3), but", img.shape)
    img = np.expand_dims(img, axis=0)

    # get features
    extracted_features = model(img).numpy().flatten().tolist()

    return extracted_features


def tf_create_queries(directory, model):
    """ Read test data and create Elasticsearch queries.

    The image queries structure is the following: ("id", "filename", "path", "features").
    The "features" field refers to the image feature vector which consists of:
        * the image embeddings found by the deep-learning model and then reduced using PCA,
        * the one-hot class label vector, where the class is predicted by the deep-learning model.

    Args:
        directory:
            test data directory, as string.
        model:
            deep-learning model, as Pytorch object.

    Returns:
        image queries, as list of dictionaries.
    """
    if not os.path.isdir(directory):
        app.logger.error(f"Provided path doesn't exist or isn't a directory ...")
        return None
    elif model is None:
        app.logger.error(f"Provided deep-learning model is None ...")
        return None

    image_filenames = os.listdir(directory)
    # get batch size for prediction
    steps = 50
    batch_size = len(image_filenames) // steps
    if len(image_filenames) % steps != 0.:
        batch_size += 1
    print("batch size =", batch_size)

    # predict over batches of images
    queries = []
    for batch in tqdm(range(batch_size), desc="batch prediction"):
        start_ind = batch * steps
        end_ind = min(batch * steps + steps, len(image_filenames))

        current_filenames = image_filenames[start_ind:end_ind]
        images = []
        for filename in current_filenames:
            img_full = plt.imread(os.path.join(directory, filename))
            img = cv2.resize(img_full, (224, 224))
            if img.shape != (224, 224, 3):
                print(img.shape, filename)
            images.append(img)
        images = np.array(images)

        extracted_features = model(images).numpy()

        for ind, filename in enumerate(current_filenames):
            filepath = directory + "/" + filename
            doc = {
                'id': filename,
                'filename': filename,
                'path': filepath,
                'features': extracted_features[ind, :].tolist()
            }
            queries.append(doc)
    print("queries len", len(queries))

    return queries


def write_results(results, path):
    """ Create search results file (.txt) according to trec_eval specifications.

    The results file has records of the form: (query_id, iteration, doc_id, rank, similarity, run_id).

    Args:
        results:
            search results of the form (query_id, images: [id, filename, path, score]), as list of dictionaries.
        path:
             file path, as string.
    """
    if (results is None) or (not results):
        app.logger.error("Number of search results is 0 ...")
        return
    elif os.path.isdir(path):
        app.logger.error("Provided path is a directory and not a file ...")
        return

    with open(path, 'w') as f:
        iteration = "0"
        rank = "0"
        run_id = "STANDARD"
        for result in tqdm(results, desc="saving results"):
            # results file contains records of the form: (query_id, iteration, doc_id, rank, similarity, run_id)
            for image in result["images"]:
                record = f"{result['query_id']} {iteration} {image['filename']} {rank} {image['score']} {run_id}\n"
                f.write(record)


if __name__ == "__main__":
    model_path = "../saved-model/EfficientNetB3-car color-81.61.h5"
    model = load_model(model_path)
    # get needed layer of model and predict features
    feature_extraction_model = keras.Model(inputs=model.input, outputs=model.layers[-4].output)

    images_path = "../static/ford/test"
    queries = tf_create_queries(images_path, feature_extraction_model)

    # read Elastic Search DB
    with open(ES_DB_PATH, "r") as file:
        images = json.load(file)
    num_features = len(images[0]["features"])

    # run Elasticsearch
    app.logger.info(f"Running Elasticsearch on {hosts} ...")
    es = Elasticsearch(hosts=hosts, timeout=60, retry_on_timeout=True,
                       http_auth=('elastic', ES_PASSWORD))

    app.logger.info(f"Creating Elasticsearch index {index_name} ...")
    # creating Elasticsearch index
    indexer = Indexer()
    indexer.create_index(es=es,
                         name=index_name,
                         number_of_shards=number_of_shards,
                         number_of_replicas=number_of_replicas,
                         num_features=num_features)

    app.logger.info(f"Indexing images ...")
    # indexing image documents
    indexer.index_images(es=es, name=index_name, images=images)

    app.logger.info(f"Searching Elasticsearch index {index_name} ...")
    searcher = Searcher()
    results = searcher.search_index(es=es, name=index_name, queries=queries, k=10)
    if (results is None) or (not results):
        app.logger.error("Number of search results is 0 ...")
        sys.exit(1)

    # write results
    path_results = "../output/test.txt"
    app.logger.info(f"Writing search results at {path_results} ...")
    write_results(results, path_results)

    # write to pickle to keep the structure
    pickle_path = "../output/test_color.pickle"
    with open(pickle_path, "wb") as file:
        pickle.dump(results, file)
    print("saved to ", pickle_path)
