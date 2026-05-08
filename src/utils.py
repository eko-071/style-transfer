import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    return image.clip(0, 1)

def save_image(tensor, path):
    img = im_convert(tensor)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(path)
