from opensearchpy import OpenSearch
from tqdm import tqdm

class OpenSearchService:
    configuration: None
    client: None
    
    def __init__(self, configuration):
        self.configuration = configuration
        host = configuration.OPENSEARCH_HOST
        auth = (configuration.OPENSEARCH_USER, configuration.OPEN_SEARCH_PASSWORD) 
        self.client = OpenSearch(hosts = host, http_auth = auth)    
    
    def Search(self, filename, features_vec):
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
        openSearchReponse = self.client.search(index=[self.configuration.index_name], body=query_body, size=10)
        openSearchrecord = {
                'query_id': filename[0: filename.find('-')],
                'images': []
            }
        for hit in openSearchReponse['hits']['hits']:
            res = {
                    'id': hit["_source"]["id"],
                    'filename': hit["_source"]["filename"],
                    'path': self.configuration.IMAGE_BUCKET + hit["_source"]["filename"],
                    'score': hit["_score"]
                }
            openSearchrecord["images"].append(res)
            openSearchResults.append(openSearchrecord)
        results = openSearchResults[0]['images']
        
        return results
    
    def reinitOpenSearch(self, images):
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': self.configuration.number_of_shards,
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
                            'dimension': self.configuration.num_features,
                        }
                    }
                }
            }
    
        if self.client.indices.exists(index=[self.configuration.index_name]):
            print(f'Elasticsearch index "{self.configuration.index_name}" already exists. Deleting ...')
            self.client.indices.delete(index=[self.configuration.index_name])
            
        self.client.indices.create(self.configuration.index_name, body=index_body)

        for image in tqdm(images, desc="indexing images in ES"):
            self.client.index(index=self.configuration.index_name, body=image)