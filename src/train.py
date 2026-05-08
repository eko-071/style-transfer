import torch
from torch import optim
import utils
import model

def run_style_transfer(content_path, style_path, steps=1000, style_weight=1e6, content_weight=1, show_every=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content_image = utils.load_image(content_path).to(device)
    style_image = utils.load_image(style_path, shape=content_image.shape[-2:]).to(device)

    vgg = model.load_vgg().to(device).eval()

    content_features = model.get_features(content_image, vgg)
    style_features = model.get_features(style_image, vgg)
    style_grams = {layer: model.gram_matrix(style_features[layer]) for layer in style_features}

    target_image = content_image.clone().requires_grad_(True).to(device)

    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }

    optimizer = optim.Adam([target_image], lr=0.003)

    for i in range(1, steps + 1):
        target_features = model.get_features(target_image, vgg)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = model.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
            
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % show_every == 0 or i == steps:
            print(f"Step {i}/{steps} | Loss: {total_loss.item():.4f}")
            utils.save_image(target_image, f"data/output/stylized_step_{i}.jpg")

if __name__ == "__main__":
    run_style_transfer('data/content/golden_gate_bridge.jpg', 'data/style/starry_night.jpg')
