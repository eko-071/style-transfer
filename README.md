# Neural Style Transfer (NST) 🎨

An elegant implementation of Neural Style Transfer using PyTorch and the VGG-19 Convolutional Neural Network. This project was developed as a course project for *Essentials of AI and ML*.

## Overview
Neural Style Transfer is a computer vision technique that generates a new image by combining the **content** of one image (e.g., a photograph) with the **artistic style** of another (e.g., a famous painting). 

The system uses a pre-trained VGG-19 neural network to extract deep feature representations. By using gradient descent, it iteratively modifies a target image to minimize two distinct loss functions simultaneously:
- **Content Loss:** Ensures the generated image retains the structure and objects of the original photo.
- **Style Loss:** Ensures the generated image adopts the texture, color palette, and brushstrokes of the style artwork (calculated via Gram Matrices).

## Repository Structure
- `data/content/`: Source photographs.
- `data/style/`: Source artwork/style images.
- `data/output/`: The generated stylized images.
- `src/cli.py`: The command-line interface for running experiments.
- `src/train.py`: The core optimization loop and loss calculations.
- `src/model.py`: VGG-19 network setup and feature extraction logic.
- `src/utils.py`: Image loading, preprocessing, and tensor de-processing.

## Setup Instructions

1. Clone the repository and navigate into it.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can generate your own stylized images using the `cli.py` script. 

### Basic Example
```bash
python src/cli.py --content data/content/landscape.jpg --style data/style/starry_night.jpg
```

### Advanced Usage (Tuning Parameters)
You can control the balance between the original photo and the artistic style using `--style_weight`. A higher style weight means the painting's style will be much more prominent.
```bash
python src/cli.py \
    --content data/content/landscape.jpg \
    --style data/style/the_scream.jpg \
    --style_weight 5e6 \
    --steps 1000 \
    --show_every 200
```

### CLI Arguments
- `--content`: Path to the content image (Required).
- `--style`: Path to the style image (Required).
- `--steps`: Total number of optimization iterations (Default: 1000).
- `--style_weight`: How strongly to apply the style (Default: 1e6).
- `--content_weight`: How strongly to preserve the content (Default: 1).
- `--show_every`: Save intermediate results every X steps (Default: 200).
