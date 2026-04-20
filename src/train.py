"""
train.py
The main script containing the optimization loop for Neural Style Transfer.
"""

import torch
from torch import optim
import utils
import model

def run_style_transfer(content_path, style_path, steps=1000, style_weight=1e6, content_weight=1, show_every=200):
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load images and model
    print(f"Loading content image from {content_path}...")
    content_image = utils.load_image(content_path).to(device)
    
    print(f"Loading style image from {style_path}...")
    # Resize style image to match the content image's shape for easier processing
    style_image = utils.load_image(style_path, shape=content_image.shape[-2:]).to(device)

    print("Loading VGG-19 model...")
    vgg = model.load_vgg().to(device).eval()

    # 3. Extract static features
    # Get content features (we only use the 'conv4_2' layer for content representation)
    content_features = model.get_features(content_image, vgg)
    
    # Get style features and compute their Gram matrices (which capture the 'texture' representation)
    style_features = model.get_features(style_image, vgg)
    style_grams = {layer: model.gram_matrix(style_features[layer]) for layer in style_features}

    # 4. Initialize target image
    # Starting with a copy of the content image makes the optimization converge much faster than random noise
    target_image = content_image.clone().requires_grad_(True).to(device)

    # 5. Define Weights and Optimizer
    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }

    # Use Adam optimizer to update the target image pixels
    optimizer = optim.Adam([target_image], lr=0.003)

    # 6. Optimization Loop    
    print("Starting optimization loop...")
    for i in range(1, steps + 1):
        # Forward pass the target image through VGG to get its current features
        target_features = model.get_features(target_image, vgg)
        
        # --- Calculate Content Loss ---
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # --- Calculate Style Loss ---
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = model.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            
            style_gram = style_grams[layer]
            
            # Mean Squared Error between the gram matrices, scaled down by layer size
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
            
        # --- Total Loss ---
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Backpropagation: compute gradients
        optimizer.zero_grad()
        total_loss.backward()
        
        # Step: update the target image pixels
        optimizer.step()
        
        # Display progress and save intermediate images
        if i % show_every == 0 or i == steps:
            print(f"Step {i}/{steps} | Total Loss: {total_loss.item():.4f}")
            utils.save_image(target_image, f"data/output/stylized_step_{i}.jpg")

    print("Finished! Final stylized image saved to data/output/.")

if __name__ == "__main__":
    # If run directly without CLI, use default paths
    run_style_transfer('data/content/golden_gate_bridge.jpg', 'data/style/starry_night.jpg')
