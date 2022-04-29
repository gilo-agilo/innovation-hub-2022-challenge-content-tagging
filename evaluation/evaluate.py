import os
import sys
import pickle
from tqdm import tqdm
import numpy as np
from flask import Flask
import logging
from PIL import Image
import joblib
from elasticsearch import Elasticsearch

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from constants import *
from models.image_dataset import ImageDataset
from models.utils import predict
from models import pretrained_models
from index.searcher import Searcher

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

ES_PASSWORD = "+b77YVyI_QDtEAMO=bRl"

hosts = "http://localhost:9200"
index_name = 'cifar10'
number_of_shards = 30
number_of_replicas = 0

# hook variable for VGG-16 image embeddings
hook_features = []


def get_features():
    """ Hook for extracting image embeddings from the layer that is attached to.

    Returns:
        hook, as callable.
    """
    def hook(model, input, output):
        global hook_features
        hook_features = output.detach().cpu().numpy()
    return hook


def create_queries(directory, model, pca, transform, num_labels):
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
        pca:
           Principal Component Analysis (PCA), as scikit-learn model.
        transform:
            image transformations, as Pytorch object.
        num_labels:
            number of class labels, as integer.

    Returns:
        image queries, as list of dictionaries.
    """
    if not os.path.isdir(directory):
        app.logger.error(f"Provided path doesn't exist or isn't a directory ...")
        return None
    elif model is None:
        app.logger.error(f"Provided deep-learning model is None ...")
        return None
    elif pca is None:
        app.logger.error(f"Provided PCA model is None ...")
        return None

    queries = []
    for file in tqdm(os.listdir(directory), desc="getting features from images"):
        path = os.path.join(directory, file)

        with Image.open(path) as image:
            # create dataset and dataloader objects for Pytorch
            dataset = ImageDataset([image], transform)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

            # pass image trough deep-learning model to gain the image embedding vector
            # and predict the class
            pred = predict(dataloader, model, device)

            # extract the image embeddings vector
            embedding = hook_features
            # reduce the dimensionality of the embedding vector
            embedding = pca.transform(embedding)

            # get image class label as one-hot vector
            label_vec = np.zeros(num_labels, dtype='int64')
            label_vec[pred] = 1

            # concatenate embeddings and label vector
            features_vec = np.concatenate((embedding, label_vec), axis=None)

            query = {
                'id': file,
                'filename': file,
                'path': path,
                'features': features_vec
            }
            queries.append(query)

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
    # get available device (CPU/GPU)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    app.logger.info(f'Using {device} device ...')

    # initialize VGG-16
    app.logger.info(f'Loading VGG-16 model from {PATH_VGG_16} ...')
    model = pretrained_models.initialize_model(pretrained=True,
                                               num_labels=len(LABEL_MAPPING),
                                               feature_extracting=True)
    # load VGG-16 pretrained weights
    # model.load_state_dict(torch.load(path_vgg_16, map_location='cuda:0'))
    model.load_state_dict(torch.load(PATH_VGG_16, map_location='cpu'))
    # send VGG-16 to CPU/GPU
    model.to(device)
    # register hook
    model.classifier[5].register_forward_hook(get_features())

    # load PCA pretrained model
    app.logger.info(f'Loading PCA model from {PATH_PCA} ...')
    pca = joblib.load(PATH_PCA)

    # image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get features of the test images
    queries = create_queries(directory="../static/ford/test",
                             model=model, pca=pca, transform=transform, num_labels=10)

    # run Elasticsearch
    app.logger.info(f"Running Elasticsearch on {hosts} ...")
    es = Elasticsearch(hosts=hosts, timeout=60, retry_on_timeout=True,
                       http_auth=('elastic', ES_PASSWORD))

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
    with open("../output/test.pickle", "wb") as file:
        pickle.dump(results, file)
    print("saved to ../output/test.pickle")
