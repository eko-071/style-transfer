"""
utils.py
Contains helper functions for loading, preprocessing, deprocessing, and displaying images.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_image(img_path, max_size=400, shape=None):
    """
    Load an image and convert it to a normalized PyTorch tensor.
    VGG19 expects images to be normalized in a specific way.
    """
    image = Image.open(img_path).convert('RGB')
    
    # Resize the image if it's too large to save memory
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # VGG-19 normalization mean and std
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    # Add a batch dimension since PyTorch models expect (batch, channels, height, width)
    # Also slice to [:3,:,:] in case an image has an alpha channel (though we convert to RGB above)
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    """
    Convert a PyTorch tensor back into a numpy image for display/saving.
    Undoes the VGG19 normalization.
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    
    # Tensor is (C, H, W). We need (H, W, C) for image formats
    image = image.transpose(1, 2, 0)
    
    # Un-normalize the image back to normal pixel ranges
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    
    # Clip values to ensure they are valid image pixel values between 0 and 1
    image = image.clip(0, 1)
    
    return image

def save_image(tensor, path):
    """
    Save a PyTorch tensor directly as an image file.
    """
    img = im_convert(tensor)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(path)
