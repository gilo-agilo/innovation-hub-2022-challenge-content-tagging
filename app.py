import os
import logging
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, jsonify
from opensearchpy import OpenSearch
from tqdm import tqdm

import joblib
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import *
from models.image_dataset import ImageDataset
from models.utils import predict
from models import pretrained_models
from index.indexer import Indexer
from index.searcher import Searcher

ES_PASSWORD = "+b77YVyI_QDtEAMO=bRl"
ES_DB_PATH = "es_db/train.json"

ProductionMode = os.getenv('elastciDn') != None
# ProductionMode = False

# FIXME
# dir_train = "static/cifar10/train"
# dir_test = "static/cifar10/test"
dir_test = "static/ford/test"

hosts = "http://localhost:9200"
index_name = 'cifar10'
number_of_shards = 30
number_of_replicas = 0

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# hook variable for VGG-16 image embeddings
hook_features = []


# FIXME: change hooks into simple output
def get_features():
    """ Hook for extracting image embeddings from the layer that is attached to.

    Returns:
        hook, as callable.
    """
    def hook(model, input, output):
        global hook_features
        hook_features = output.detach().cpu().numpy()
    return hook


@app.route('/')
def load_page():
    """ Render index.html webpage. """
    return render_template('index.html')



@app.route('/ImageVector', methods=['POST'])
def imageVector():
    image = Image.open(request.files['image-file'].stream)

    features_vec = imageVectorInternal(image)

    return json.dumps({
        "vector" : features_vec.tolist()
    })

def imageVectorInternal(image):
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

    # get image clas s label as one-hot vector
    label_vec = np.zeros(len(LABEL_MAPPING), dtype='int64')
    label_vec[pred] = 1

    # concatenate embeddings and label vector
    features_vec = np.concatenate((embedding, label_vec), axis=None)
    
    return features_vec

def renderTemplate(image, results, pageName):
    # prepare image for html
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode()

    # prepare list with all input image info
    input_img = "data:image/png;base64, " + img_str

    return render_template(pageName, results=results,
                           input_img=input_img, input_img_filename=request.files['image-file'].filename)

@app.route('/', methods=['POST'])
def search():
    image = Image.open(request.files['image-file'].stream)
    features_vec = imageVectorInternal(image);

    filename = request.files['image-file'].filename
    query = {
        'id': filename,
        'filename': filename,
        'path': os.path.join(dir_test, filename),
        'features': features_vec
    }

    results = searcher.search_index(es=es, name=index_name, queries=[query], k=10)
    results = results[0]['images']

    
    return renderTemplate(image, results, 'index.html')

@app.route('/searchOpenSearch', methods=['POST'])
def searchOpenSearch():
    image = Image.open(request.files['image-file'].stream)
    features_vec = imageVectorInternal(image);

    filename = request.files['image-file'].filename
    query_body = {
        "query": {
            "knn" : {
                "features" :{
                    "vector" : features_vec,
                    "k": 10
                }
            } 
        }
    }
    
    openSearchResults = []
    openSearchReponse = client.search(index=[index_name], body=query_body, size=10)
    openSearchrecord = {
                'query_id': filename[0: filename.find('-')],
                'images': []
            }
    for hit in openSearchReponse['hits']['hits']:
        res = {
                'id': hit["_source"]["id"],
                'filename': hit["_source"]["filename"],
                'path': hit["_source"]["path"],
                'score': hit["_score"]
            }
        openSearchrecord["images"].append(res)
        openSearchResults.append(openSearchrecord)
    results = openSearchResults[0]['images']
    
    return renderTemplate(image, results, 'index-opensearch.html')

@app.route('/sarchOpenSearch')
def load_page_OpenSearch():
    """ Render index.html webpage. """
    return render_template('index-opensearch.html')

@app.route('/reinitOpenSearch')
def reinitOpenSearch():
    index_body = {
        'settings': {
            'index': {
                'number_of_shards': number_of_shards,
                'knn': True
            }
        },
        'mappings': {
                'properties': {
                    'id': {
                        'type': 'text',
                        'index': False
                    },
                    'filename': {
                        'type': 'text',
                        'index': False
                    },
                    'path': {
                        'type': 'text',
                        'index': False
                    },
                    'features': {
                        'type': 'knn_vector',
                        'dimension': num_features,
                    }
                }
            }
    }
    
    if client.indices.exists(index=[index_name]):
        print(f'Elasticsearch index "{index_name}" already exists. Deleting ...')
        client.indices.delete(index=[index_name])
        
    client.indices.create(index_name, body=index_body)

    for image in tqdm(images, desc="indexing images in ES"):
        client.index(index=index_name, body=image)
    return "ok"

if __name__ == '__main__':
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

    # read Elastic Search DB
    with open(ES_DB_PATH, "r") as file:
        images = json.load(file)
    num_features = len(images[0]["features"])

    app.logger.info(f"Running Elasticsearch on {hosts} ...")
    # run Elasticsearch
    es = Elasticsearch(hosts=hosts, timeout=60, retry_on_timeout=True,
                       http_auth=('elastic', ES_PASSWORD))

    if ProductionMode:
        es = Elasticsearch(hosts='http://' + os.environ['elastciDn'] + ':9200')
    else: 
        #es = Elasticsearch(hosts='http://localhost:30002')
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

    searcher = Searcher()
    
    host = 'https://search-testdomain-6ymb6zjdmjqxog7kln72dpya7m.eu-west-1.es.amazonaws.com'
    auth = ('testdomainUser', 'Qwerty1234!') # For testing only. Don't store credentials in code.
    client = OpenSearch(hosts = host, http_auth = auth)    

    if ProductionMode:
        app.logger.info("Running application Production mode...")
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        app.logger.info("Running application local mode...")
        app.run()


