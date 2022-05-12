from models import pretrained_models
import joblib
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from constants import *
from models.utils import predict
import numpy as np
from models.image_dataset import ImageDataset

class AIImageService:
    hook_features = []
    app = None
    pca = None
    model = None
    transform = None
    device = None
    
    def __init__(self, app):
        self.app = app
    
    def get_features(self):
        """ Hook for extracting image embeddings from the layer that is attached to.

        Returns:
            hook, as callable.
        """
        def hook(model, input, output):
            global hook_features
            self.hook_features = output.detach().cpu().numpy()
        return hook
    
    def imageVectorInternal(self, image):
        # create dataset and dataloader objects for Pytorch
        dataset = ImageDataset([image], self.transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        # pass image trough deep-learning model to gain the image embedding vector
        # and predict the class
        pred = predict(dataloader, self.model, self.device)

        # extract the image embeddings vector
        embedding = self.hook_features
        # reduce the dimensionality of the embedding vector
        embedding = self.pca.transform(embedding)

        # get image clas s label as one-hot vector
        label_vec = np.zeros(len(LABEL_MAPPING), dtype='int64')
        label_vec[pred] = 1

        # concatenate embeddings and label vector
        features_vec = np.concatenate((embedding, label_vec), axis=None)
        
        return features_vec
    
    
    def init(self):
        # get available device (CPU/GPU)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.app.logger.info(f'Using {self.device} device ...')

        # initialize VGG-16
        self.app.logger.info(f'Loading VGG-16 model from {PATH_VGG_16} ...')
        self.model = pretrained_models.initialize_model(pretrained=True,
                                               num_labels=len(LABEL_MAPPING),
                                               feature_extracting=True)
        # load VGG-16 pretrained weights
        # model.load_state_dict(torch.load(path_vgg_16, map_location='cuda:0'))
        self.model.load_state_dict(torch.load(PATH_VGG_16, map_location='cpu'))
        # send VGG-16 to CPU/GPU
        self.model.to(self.device)
        # register hook
        self.model.classifier[5].register_forward_hook(self.get_features())
        
        # load PCA pretrained model
        self.app.logger.info(f'Loading PCA model from {PATH_PCA} ...')
        self.pca = joblib.load(PATH_PCA)

        # image transformations
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])