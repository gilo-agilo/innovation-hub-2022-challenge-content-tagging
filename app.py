import logging
import json
from PIL import Image
from flask import Flask, render_template, request

from configuration import Configuration
from services.aiImageService import AIImageService
from services.elasticSearchService import ElasticSearchService
from services.imageService import ImageService
from services.openSearchService import OpenSearchService
import os

from services.templateService import TemplateService

ProductionMode = os.getenv('elastciDn') != None

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

conf = Configuration()
service = AIImageService(app)
imageService = ImageService(conf)
openSearchService = OpenSearchService(conf)
elasticSerchService = ElasticSearchService(app, conf, ProductionMode)
templateService = TemplateService(imageService)

@app.route('/')
def load_page():
    return render_template('index.html')

@app.route('/searchOpenSearch')
def load_page_OpenSearch():
    return render_template('index-opensearch.html')

@app.route('/', methods=['POST'])
def search():
    image = Image.open(request.files['image-file'].stream)
    features_vec = service.imageVectorInternal(image);
    filename = request.files['image-file'].filename
    results = elasticSerchService.Search(filename, features_vec)

    return templateService.renderTemplate(image, results, 'index.html')

@app.route('/searchOpenSearch', methods=['POST'])
def searchOpenSearch():
    image = Image.open(request.files['image-file'].stream)
    features_vec =  service.imageVectorInternal(image);
    filename = request.files['image-file'].filename
    results = openSearchService.Search(filename, features_vec)
    
    return templateService.renderTemplate(image, results, 'index-opensearch.html')

@app.route('/ImageVector', methods=['POST'])
def imageVector():
    image = Image.open(request.files['image-file'].stream)

    features_vec =  service.imageVectorInternal(image)

    return json.dumps({
        "vector" : features_vec.tolist()
    })

@app.route('/reinitOpenSearch')
def reinitOpenSearch():
    openSearchService.reinitOpenSearch(imageService.images)
    
    return "ok"

if __name__ == '__main__':
    service.init()   
    imageService.InitImages()
    elasticSerchService.init(imageService.images)
    
    if ProductionMode:
        app.logger.info("Running application Production mode...")
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        app.logger.info("Running application local mode...")
        app.run()