"""
model.py
Contains the CNN (VGG-19) setup, feature extraction logic, and the definitions 
for Content Loss, Style Loss, and Total Variation Loss.
"""

import torch
from torchvision import models

def load_vgg():
    """
    Loads the pre-trained VGG-19 model and freezes its parameters.
    We only use the 'features' portion of the model (convolutional and pooling layers),
    and ignore the fully connected classification layers at the end.
    """
    # Load VGG-19 features part
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    
    # Freeze all VGG parameters since we're only optimizing the target image pixels, not the model weights
    for param in vgg.parameters():
        param.requires_grad = False
        
    return vgg

def get_features(image, model, layers=None):
    """
    Run an image forward through a model and extract the features for a specific set of layers.
    """
    # These represent the exact indices of the layers in the VGG19 features module.
    # We map them to human-readable names based on Gatys et al. paper.
    if layers is None:
        layers = {
            '0': 'conv1_1',  # Style feature
            '5': 'conv2_1',  # Style feature
            '10': 'conv3_1', # Style feature
            '19': 'conv4_1', # Style feature
            '21': 'conv4_2', # Content representation
            '28': 'conv5_1'  # Style feature
        }
        
    features = {}
    x = image
    # Iterate through all the layers in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """
    Calculate the Gram Matrix of a given tensor.
    The Gram Matrix measures the correlation between different feature maps, 
    effectively capturing the 'style' or texture of the image regardless of spatial layout.
    """
    # Get batch_size (b), depth (d), height (h), width (w)
    b, d, h, w = tensor.size()
    
    # Reshape the tensor so we are multiplying the features for each channel
    tensor = tensor.view(b * d, h * w)
    
    # Calculate the gram matrix (multiply tensor by its transpose)
    gram = torch.mm(tensor, tensor.t())
    
    return gram
