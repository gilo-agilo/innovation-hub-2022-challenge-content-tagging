from elasticsearch import Elasticsearch
import os

from index.indexer import Indexer
from index.searcher import Searcher

class ElasticSearchService:
    app = None
    es = None
    searcher = None
    configuration = None
    
    def __init__(self, app, configuration, productionMode):
        self.app = app
        self.configuration=configuration
        app.logger.info(f"Running Elasticsearch on {configuration.ES_HOST} ...")
        # run Elasticsearch
        self.es = Elasticsearch(hosts=configuration.ES_HOST, timeout=60, retry_on_timeout=True,
                       http_auth=('elastic', configuration.ES_PASSWORD))

        if productionMode:
            self.es = Elasticsearch(hosts='http://' + os.environ['elasticDns'] + ':9200')
        else: 
            #self.es = Elasticsearch(hosts='http://localhost:30002')
            es = Elasticsearch(hosts=configuration.ES_HOST, timeout=60, retry_on_timeout=True,
                          http_auth=('elastic', configuration.ES_PASSWORD))
            
    def init(self, images):
        num_features = len(images[0]["features"])

        self.app.logger.info(f"Creating Elasticsearch index {self.configuration.INDEX_NAME} ...")
        # creating Elasticsearch index
        indexer = Indexer()
        indexer.create_index(es=self.es,
                            name=self.configuration.INDEX_NAME,
                            number_of_shards=self.configuration.ES_NUMBER_OF_SHARDS,
                            number_of_replicas=self.configuration.ES_NUMBER_OF_REPLICAS,
                            num_features=num_features)

        self.app.logger.info(f"Indexing images ...")
        # indexing image documents
        indexer.index_images(es=self.es, name=self.configuration.INDEX_NAME, images=images)

        self.searcher = Searcher()
        
    def Search(self, filename, features_vec):
        query = {
            'id': filename,
            'filename': filename,
            'path': os.path.join(self.configuration.DIR_DESTINATION, filename),
            'features': features_vec
        }

        results = self.searcher.search_index(es=self.es, name=self.configuration.INDEX_NAME, queries=[query], k=10)
        results = results[0]['images']
        
        return results
        