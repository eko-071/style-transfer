# Neural Style Transfer

This project provides a PyTorch implementation of Neural Style Transfer. It uses a pre-trained VGG-19 model to combine the content of one image with the artistic style of another.

## Overview

Neural Style Transfer works by taking two images, a content image (like a photograph) and a style image (like a painting), and blending them together. The system optimizes a target image to match the high-level structure of the content image and the textures and colors of the style image.

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Use the provided `cli.py` script to generate your own stylized images.

### Basic Example

```bash
python src/cli.py --content data/content/landscape.jpg --style data/style/starry_night.jpg
```

### Advanced Usage

You can tweak the parameters to control the output. For example, to make the style more prominent, increase the `--style_weight`:

```bash
python src/cli.py --content data/content/landscape.jpg --style data/style/starry_night.jpg --style_weight 5e6 --steps 1000
```

### Options

- `--content`: Path to the content image (Required)
- `--style`: Path to the style image (Required)
- `--steps`: Number of optimization steps (Default: 1000)
- `--style_weight`: Weight for the style loss (Default: 1e6)
- `--content_weight`: Weight for the content loss (Default: 1)
- `--show_every`: Save an intermediate image every X steps (Default: 200)
