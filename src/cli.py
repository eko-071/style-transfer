"""
cli.py
Command Line Interface for running Neural Style Transfer with custom arguments.
"""

import argparse
from train import run_style_transfer

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer CLI")
    
    parser.add_argument('--content', type=str, required=True, help="Path to the content image")
    parser.add_argument('--style', type=str, required=True, help="Path to the style image")
    parser.add_argument('--steps', type=int, default=1000, help="Number of optimization steps (default: 1000)")
    parser.add_argument('--style_weight', type=float, default=1e6, help="Weight for the style loss (default: 1e6)")
    parser.add_argument('--content_weight', type=float, default=1, help="Weight for the content loss (default: 1)")
    parser.add_argument('--show_every', type=int, default=200, help="Save an intermediate image every X steps (default: 200)")

    args = parser.parse_args()

    print("--- Neural Style Transfer ---")
    print(f"Content Image: {args.content}")
    print(f"Style Image:   {args.style}")
    print(f"Steps:         {args.steps}")
    print(f"Style Weight:  {args.style_weight}")
    print("-----------------------------")

    # Call the core logic function from train.py
    run_style_transfer(
        content_path=args.content,
        style_path=args.style,
        steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        show_every=args.show_every
    )

if __name__ == "__main__":
    main()
