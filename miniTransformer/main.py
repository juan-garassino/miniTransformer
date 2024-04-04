"""Main module for the miniTransformer package execution.

This script allows for training the miniTransformer model or generating text from a trained model.
"""

import argparse
import os
import sys
import torch
from miniTransformer.fitting.train import train, generate_text, BigramLanguageModel
from miniTransformer.preprocessing.sourcing.sourcing import (
    load_data,
    create_char_mappings,
)

# Assuming create_animation is used elsewhere or will be used in future updates.

def parse_arguments():
    """Parse command line arguments for the miniTransformer script."""
    parser = argparse.ArgumentParser(description="Train a miniTransformer model.")
    parser.add_argument("--root_dir", type=str, default="Code/juan-garassino")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--colab", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/miniTransformer/data")
    parser.add_argument("--name", type=str, default="input.txt")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--checkpoints_dir", type=str, default="/miniTransformer/results/checkpoints"
    )
    parser.add_argument(
        "--heatmaps_dir", type=str, default="/miniTransformer/results/heatmaps"
    )
    parser.add_argument(
        "--animations_dir", type=str, default="/miniTransformer/results/animations"
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--heatmap_interval",
        type=int,
        default=100,
        help="Interval between saving attention heatmaps.",
    )
    parser.add_argument("--n_of_char", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Adjust directories based on whether running in Colab or not

    args.data_dir = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.data_dir.lstrip("/")
    )
    args.checkpoints_dir = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.checkpoints_dir.lstrip("/")
    )
    args.heatmaps_dir = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.heatmaps_dir.lstrip("/")
    )
    args.animations_dir = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.animations_dir.lstrip("/")
    )

    if args.generate:
        if args.checkpoint:
            device = torch.device(args.device)
            checkpoint_path = os.path.join(args.checkpoints_dir, args.checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_state_dict = checkpoint["model_state_dict"]
            data = load_data(args.data_dir)
            char_to_int, int_to_char, vocab_size = create_char_mappings(data)
            model = BigramLanguageModel(
                vocab_size,
                args.n_embd,
                args.block_size,
                args.n_head,
                args.n_layer,
                args.dropout,
            ).to(device)
            model.load_state_dict(model_state_dict)
            model.eval()
            print("\nGenerating text:\n")
            for char in generate_text(
                model, int_to_char, device, max_new_tokens=args.n_of_char
            ):
                print(char, end="", flush=True)
        else:
            print("Please provide a checkpoint file to generate text.", file=sys.stderr)
    else:
        train(
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            learning_rate=args.learning_rate,
            device=args.device,
            eval_iters=args.eval_iters,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            colab=args.colab,
            path=args.data_dir,
            name=args.name,
            heatmap_interval=args.heatmap_interval,
            save_interval=args.save_interval,
            checkpoints_dir=args.checkpoints_dir,
            heatmaps_dir=args.heatmaps_dir,
            animations_dir=args.animations_dir,
        )
