from copy import deepcopy
import pandas as pd
import os
import pickle
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_model.VGG16_model import initialize_model, fit, predict

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    num_labels = 10
    feature_extracting = True
    pretrained = True

    # define model
    model = initialize_model(num_labels=num_labels,
                             feature_extracting=feature_extracting,
                             pretrained=pretrained).to(device)

    print(f'Model architecture:\n{model}')

    # Create an optimizer that only updates the desired parameters
    learning_rate = 0.001
    weight_decay = 0.001

    print("Parameters to learn:")
    if feature_extracting:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        params_to_update = model.parameters()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # optimizer
    optimizer = AdamW(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min')
    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # prepare datasets
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = CustomImageDataset("../data/car_models.csv",
                                  r"C:\Users\ann\Code\challenges\datasets\crawler_outside",
                                  transform=transform)
    testset = CustomImageDataset("../data/car_models_test.csv",
                                 r"C:\Users\ann\Code\challenges\datasets\ford",
                                 transform=transform)

    # fit and predict
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # best iteration
    best_epoch = 0
    best_train_loss = 0
    best_test_loss = 0
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_train_pred = None
    best_test_pred = None
    best_train_y = None
    best_test_y = None
    best_model_state_dict = None

    epochs = 1
    batch_size = 64

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        fit(train_dataloader, model, loss_fn, optimizer, print_loss=True)

        # print('\nTrain:\n-------')
        # train_current_loss, train_pred, train_y = predict(train_dataloader, model, loss_fn)
        #
        # train_acc = 100 * accuracy_score(train_y, train_pred)
        # train_loss.append(train_current_loss)
        # train_accuracy.append(train_acc)
        #
        # print(f"accuracy: {(100 * accuracy_score(train_y, train_pred)):>0.1f}%, avg loss: {train_current_loss:>8f}")

        print('\nTest:\n-------')
        test_current_loss, test_pred, test_y = predict(test_dataloader, model, loss_fn)

        test_acc = 100 * accuracy_score(test_y, test_pred)
        test_loss.append(test_current_loss)
        test_accuracy.append(test_acc)

        print(f"accuracy: {(100 * accuracy_score(test_y, test_pred)):>0.1f}%, avg loss: {test_current_loss:>8f}")

        # best iteration
        if test_acc > best_test_accuracy:
            best_epoch = t + 1
            # best_train_loss = train_current_loss
            best_test_loss = test_current_loss
            # best_train_accuracy = train_acc
            best_test_accuracy = test_acc
            # best_train_pred = train_pred
            best_test_pred = test_pred
            # best_train_y = train_y
            best_test_y = test_y
            if torch.cuda.is_available():
                model.to(torch.device("cpu"))
                best_model_state_dict = deepcopy(model.state_dict())
                model.to(device)
            else:
                best_model_state_dict = deepcopy(model.state_dict())

        scheduler.step(test_current_loss)

        print(f"\n-------------------------------")

    # save trained model
    save_path = '../saved-model/vgg16-car-models.pth'
    torch.save(best_model_state_dict, save_path)
    print("trained VGG16 model was saved to", save_path)

    # save model's parameters
    parameters_dict = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "best_train_accuracy": best_train_accuracy,
        "best_test_accuracy": best_test_accuracy,
        "best_train_pred": best_train_pred,
        "best_test_pred": best_test_pred,
        "best_train_y": best_train_y,
        "best_test_y": best_test_y,
        "best_model_state_dict": best_model_state_dict
    }
    save_path = "../saved-model/vgg16-car-models-parameters.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(parameters_dict, file)
    print("model's parameters were saved to", save_path)
