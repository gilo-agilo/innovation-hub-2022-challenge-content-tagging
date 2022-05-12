from flask import Flask, render_template, request

class TemplateService:
    imageService = None
    def __init__(self, imageService):
        self.imageService = imageService
    
    def renderTemplate(self, image, results, pageName):
        input_img = self.imageService.ImageToBase64(image)
        input_img_filename = request.files['image-file'].filename
        
        return render_template(pageName, 
                            results=results,
                            input_img=input_img, 
                            input_img_filename=input_img_filename)