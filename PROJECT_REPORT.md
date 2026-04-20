# Project Report: Neural Style Transfer
**Course:** Essentials of AI and ML

## 1. Introduction
This project focuses on the implementation of Neural Style Transfer (NST), an advanced computer vision technique that algorithmically combines the "content" of one image with the "style" of another. The objective of this project is to demonstrate how deep hierarchical features extracted from Convolutional Neural Networks (CNNs) can be separated into abstract representations of structure and texture, allowing for the generation of novel, artistic imagery.

## 2. Literature Review
The foundational concept for this project stems from the 2015 paper *"A Neural Algorithm of Artistic Style"* by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Prior to this work, transferring style between images was an extremely difficult problem requiring complex, hand-crafted algorithms. Gatys et al. demonstrated that the feature representations learned by a CNN trained for object recognition (specifically VGG-19) contain remarkably clean separations of image content and image style, revolutionizing the field of computational art and generative deep learning.

## 3. Problem Statement and Theory
### 3.1 Problem Statement
The challenge is to generate a new image that simultaneously matches the structural layout of a content image (e.g., a photograph) and the artistic texture and color palette of a style image (e.g., a painting). This requires defining mathematical metrics for both "content" and "style" so an algorithm can minimize the difference between the generated output and the inputs.

### 3.2 Underlying Theory
The system relies on **VGG-19**, a deep CNN pre-trained on the ImageNet dataset. We freeze the network weights and instead use gradient descent to optimize the actual pixel values of a generated image.

**1. Content Representation:**
Deep layers in the CNN (such as `conv4_2`) respond to high-level features like eyes, buildings, or shapes, rather than exact pixel values. We define Content Loss as the Mean Squared Error (MSE) between the deep feature maps of the generated image and the content image.

**2. Style Representation:**
To capture style (brushstrokes, textures), we look at multiple layers (`conv1_1` through `conv5_1`). We compute a **Gram Matrix**, which is the dot product of the flattened feature maps. This calculates the correlation between different features, effectively capturing texture while completely ignoring spatial structure. The Style Loss is the MSE between the Gram matrices of the generated image and the style image.

**3. Optimization:**
The Total Loss is a weighted sum: $L_{total} = \alpha L_{content} + \beta L_{style}$. By backpropagating this loss, we iteratively update the pixels of the generated image using the Adam Optimizer.

## 4. Dataset Description
Because NST is an optimization process run on single pairs of images rather than a model trained on thousands of samples, our "dataset" consists of carefully selected high-resolution source images:
- **Content Images:** Original photographs capturing specific scenes.
  - `golden_gate_bridge.jpg`: A structural landscape photo.
  - `labrador.jpg`: A detailed portrait of a dog.
  - `landscape.jpg`: A general nature scene.
- **Style Images:** Famous classical artworks providing distinct color palettes and textures.
  - `starry_night.jpg`: Vincent van Gogh (heavy, swirling brushstrokes).
  - `the_scream.jpg`: Edvard Munch (flowing, surreal colors).
  - `great_wave_off_kanagawa.jpg`: Hokusai (distinct woodblock print lines).

## 5. Results
The model was run using the Adam optimizer for 1,000 steps per image pair. A high style weight (`1e7`) was used to ensure prominent artistic effects. The optimization successfully converged, progressively blending the style and content.

### Experiment 1: Golden Gate Bridge + The Starry Night
The linear structure of the bridge was successfully preserved while the sky and water adopted Van Gogh's signature swirling brushstrokes.
**Final Output (Step 1000):**
![Golden Gate Bridge in Starry Night Style](data/output/bridge_step_1000.jpg)

### Experiment 2: Labrador + The Scream
Applying a heavy style weight to a portrait resulted in the dog's fur blending with the surreal, flowing color palette of The Scream.
**Final Output (Step 1000):**
![Labrador in The Scream Style](data/output/dog_step_1000.jpg)

### Experiment 3: Landscape + The Great Wave
The structural lines of the landscape were maintained while the texture shifted to mimic a traditional Japanese woodblock print.
**Final Output (Step 1000):**
![Landscape in Great Wave Style](data/output/land_step_1000.jpg)

## 6. Conclusion
This project successfully validated the theory that deep neural networks learn hierarchical representations where content and style are mathematically separable. By leveraging the VGG-19 architecture and optimizing pixel values via Gram Matrices, we successfully generated high-quality artistic images. Future work could involve implementing Total Variation (TV) loss to reduce high-frequency noise or utilizing feed-forward networks (like Fast Neural Style Transfer) to generate images in real-time.
