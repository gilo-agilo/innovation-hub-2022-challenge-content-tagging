from io import BytesIO
import base64

class ImageService:
    def ImageToBase64(self, image): 
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue())
        img_str = img_bytes.decode()
        input_img = "data:image/png;base64, " + img_str
        
        return input_img