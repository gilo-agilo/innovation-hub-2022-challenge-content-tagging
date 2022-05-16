import torch
from torch import nn
import torchvision.models as models

from tqdm import tqdm


def set_parameter_requires_grad(model, feature_extracting):
    """ This helper function sets the .requires_grad attribute of the parameters in the model
    to False when we are feature extracting.

    When we are feature extracting and only want to compute gradients for the newly initialized layer,
    then we want all of the other parameters to not require gradients.

    Args:
        model:
            deep learning model, as pytorch object.
        feature_extracting:
            whether or not we're feature extracting, as boolean.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_labels, feature_extracting, pretrained=True):
    """ Initialize VGG-16 model and reshape the last layer with the correct number of classes.

    Since VGG-16 has been pretrained on Imagenet, it has output layers of size 1000, one node for each class.
    We reshape the last layer to have the same number of inputs as before, and to have the same number of
    outputs as the number of classes in our the dataset.

    Args:
        num_labels:
            number of labels in our dataset, as integer.
        feature_extracting:
          flag for feature extracting (when False, we finetune the whole model,
          when True we only update the reshaped layer params), as boolean.
        pretrained:
            whether or not we want the pretrained version of AlexNet, as boolean.

    Returns:
        VGG-16 model, as pytorch object
    """
    model = models.vgg16(pretrained=pretrained)

    set_parameter_requires_grad(model, feature_extracting)

    last_layer_in_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(last_layer_in_ftrs, num_labels)

    return model


def fit(dataloader, model, loss_fn, optimizer,
        print_loss=False, device="cpu"):
    """ Fit deep learning model.

    Args:
        dataloader:
            pytorch DataLoader object.
        model:
            deep learning model, as pytorch object.
        loss_fn:
            loss function, as pytorch object.
        optimizer:
            optimizer function, as pytorch object.
        print_loss:
            print loss on every batch, as boolean (default False)
    """
    size = len(dataloader.dataset)
    model.train()  # put on train mode
    for batch, (X, Y) in tqdm(enumerate(dataloader), desc="fitting"):
        X, Y = X.to(device), Y.to(device)

        # compute prediction
        pred = model(X)

        # compute loss
        loss = loss_fn(pred, Y)

        # reset the gradients
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update parameters
        optimizer.step()

        if print_loss and batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def predict(dataloader, model, loss_fn, device="cpu"):
    """ Predict with deep learning model.

    Args:
        dataloader:
            pytorch DataLoader object.
        model:
            deep learning model, as pytorch object.
        loss_fn:
            loss function, as pytorch object.

    Returns:
         test loss, as float.
         predictions, as a list of integers.
         ground truth, as a list of integers.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0

    pred_concat = []
    y_concat = []

    model.eval()  # put on evaluation mode
    with torch.no_grad():
        for X, Y in tqdm(dataloader, desc="evaluating"):
            X, Y = X.to(device), Y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, Y).item()

            # predictions to one-hot vectors
            for label in pred.argmax(1):
                pred_concat.append(label.item())

            # ground truth to one-hot vectors
            for label in Y:
                y_concat.append(label.item())

    test_loss /= num_batches

    return test_loss, pred_concat, y_concat

