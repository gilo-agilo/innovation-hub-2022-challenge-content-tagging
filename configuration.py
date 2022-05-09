import os

class Configuration:
    ES_PASSWORD = "+b77YVyI_QDtEAMO=bRl"
    DB_INIT_FILE = "https://data-science-cars-images.s3.eu-west-2.amazonaws.com/data/train.json"
    IMAGE_BUCKET = "https://data-science-cars-images.s3.eu-west-2.amazonaws.com/"
    OPENSEARCH_HOST = "https://search-testdomain-6ymb6zjdmjqxog7kln72dpya7m.eu-west-1.es.amazonaws.com"
    OPENSEARCH_USER = "testdomainUser"
    OPEN_SEARCH_PASSWORD = "Qwerty1234"
    
    INDEX_NAME = "cifar10"
    
    ES_NUMBER_OF_SHARDS = 30
    ES_NUMBER_OF_REPLICAS = 0
    ES_HOST = "http://localhost:9200"
    
    DIR_DESTINATION = "dir_test"
    
    def __init__(self, production_mode):
        if (production_mode):
            self.OPENSEARCH_USER = os.getenv('OPENSEARCH_USER')
            self.OPEN_SEARCH_PASSWORD = os.getenv('OPEN_SEARCH_PASSWORD')
            self.OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST')