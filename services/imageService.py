from io import BytesIO
import base64
import urllib.request
import json

class ImageService:
    configuration = None
    images = []
    
    def __init__(self, configuration):
        self.configuration = configuration
        
    def ImageToBase64(self, image): 
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue())
        img_str = img_bytes.decode()
        input_img = "data:image/png;base64, " + img_str
        
        return input_img
    
    def InitImages(self):
        with urllib.request.urlopen(self.configuration.DB_INIT_FILE) as url:
            self.images = json.loads(url.read())