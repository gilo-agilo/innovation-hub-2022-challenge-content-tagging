import os
import logging
import json
import base64
from io import BytesIO
from pydoc import importfile
from PIL import Image
import numpy as np
from elasticsearch import Elasticsearch
from flask import Flask, render_template, request, jsonify
from opensearchpy import OpenSearch
from tqdm import tqdm
import urllib.request

from constants import *
from models.image_dataset import ImageDataset
from models import pretrained_models
from index.indexer import Indexer
from index.searcher import Searcher
from configuration import Configuration
from services.aiImageService import AIImageService

conf = Configuration()

ES_PASSWORD = conf.ES_PASSWORD
DB_INIT_FILE = conf.DB_INIT_FILE
IMAGE_BUCKET = conf.IMAGE_BUCKET

ProductionMode = os.getenv('elastciDn') != None

dir_test = conf.DIR_DESTINATION
hosts = conf.ES_HOST
index_name = conf.INDEX_NAME
number_of_shards = conf.ES_NUMBER_OF_SHARDS
number_of_replicas = conf.ES_NUMBER_OF_REPLICAS

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

service = None

@app.route('/')
def load_page():
    """ Render index.html webpage. """
    return render_template('index.html')

@app.route('/ImageVector', methods=['POST'])
def imageVector():
    image = Image.open(request.files['image-file'].stream)

    features_vec =  service.imageVectorInternal(image)

    return json.dumps({
        "vector" : features_vec.tolist()
    })

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
    features_vec = service.imageVectorInternal(image);

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
    features_vec =  service.imageVectorInternal(image);

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
                'path': IMAGE_BUCKET + hit["_source"]["filename"],
                'score': hit["_score"]
            }
        openSearchrecord["images"].append(res)
        openSearchResults.append(openSearchrecord)
    results = openSearchResults[0]['images']
    
    return renderTemplate(image, results, 'index-opensearch.html')

@app.route('/searchOpenSearch')
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
    service = AIImageService(app) 
    service.init()   

    with urllib.request.urlopen(DB_INIT_FILE) as url:
        images = json.loads(url.read())
    
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
    
    host = conf.OPENSEARCH_HOST
    auth = (conf.OPENSEARCH_USER, conf.OPEN_SEARCH_PASSWORD) 
    client = OpenSearch(hosts = host, http_auth = auth)    

    if ProductionMode:
        app.logger.info("Running application Production mode...")
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        app.logger.info("Running application local mode...")
        app.run()


