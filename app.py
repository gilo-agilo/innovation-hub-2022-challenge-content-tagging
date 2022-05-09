
import logging
import json
import base64
from io import BytesIO
from pydoc import importfile
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify
from tqdm import tqdm
from constants import *
from models.image_dataset import ImageDataset
from models import pretrained_models
from index.indexer import Indexer
from configuration import Configuration
from services.aiImageService import AIImageService
from services.elasticSearchService import ElasticSearchService
from services.imageService import ImageService
from services.openSearchService import OpenSearchService
import os

conf = Configuration()

ES_PASSWORD = conf.ES_PASSWORD
DB_INIT_FILE = conf.DB_INIT_FILE
IMAGE_BUCKET = conf.IMAGE_BUCKET

ProductionMode = os.getenv('elastciDn') != None

hosts = conf.ES_HOST
index_name = conf.INDEX_NAME
number_of_shards = conf.ES_NUMBER_OF_SHARDS
number_of_replicas = conf.ES_NUMBER_OF_REPLICAS

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

service = AIImageService(app)
imageService = ImageService()
openSearchService = OpenSearchService(conf)
elasticSerchService = ElasticSearchService(app, conf, ProductionMode)

@app.route('/')
def load_page():
    return render_template('index.html')

@app.route('/ImageVector', methods=['POST'])
def imageVector():
    image = Image.open(request.files['image-file'].stream)

    features_vec =  service.imageVectorInternal(image)

    return json.dumps({
        "vector" : features_vec.tolist()
    })

def renderTemplate(image, results, pageName):
    input_img = imageService.ImageToBase64(image)
    input_img_filename = request.files['image-file'].filename
    
    return render_template(pageName, 
                           results=results,
                           input_img=input_img, 
                           input_img_filename=input_img_filename)

@app.route('/', methods=['POST'])
def search():
    image = Image.open(request.files['image-file'].stream)
    features_vec = service.imageVectorInternal(image);
    filename = request.files['image-file'].filename
    results = elasticSerchService.Search(filename, features_vec)

    return renderTemplate(image, results, 'index.html')

@app.route('/searchOpenSearch', methods=['POST'])
def searchOpenSearch():
    image = Image.open(request.files['image-file'].stream)
    features_vec =  service.imageVectorInternal(image);

    filename = request.files['image-file'].filename
    results = openSearchService.Search(filename, features_vec)
    
    return renderTemplate(image, results, 'index-opensearch.html')

@app.route('/searchOpenSearch')
def load_page_OpenSearch():
    return render_template('index-opensearch.html')

@app.route('/reinitOpenSearch')
def reinitOpenSearch():
    openSearchService.reinitOpenSearch(images)
    
    return "ok"

if __name__ == '__main__':
    service.init()   
    elasticSerchService.init()
    
    if ProductionMode:
        app.logger.info("Running application Production mode...")
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        app.logger.info("Running application local mode...")
        app.run()


