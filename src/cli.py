import argparse
from train import run_style_transfer

def main():
    parser = argparse.ArgumentParser(description="Run Neural Style Transfer on two images.")
    
    parser.add_argument('--content', required=True, help="Path to the content image")
    parser.add_argument('--style', required=True, help="Path to the style image")
    parser.add_argument('--steps', type=int, default=1000, help="Number of optimization steps")
    parser.add_argument('--style_weight', type=float, default=1e6, help="Weight for the style loss")
    parser.add_argument('--content_weight', type=float, default=1, help="Weight for the content loss")
    parser.add_argument('--show_every', type=int, default=200, help="Save intermediate results every X steps")

    args = parser.parse_args()

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
