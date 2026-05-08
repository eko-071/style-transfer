import torch
from torchvision import models

def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    return torch.mm(tensor, tensor.t())
