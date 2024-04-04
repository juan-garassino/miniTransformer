import argparse
from miniTransformer.training.train import train, generate_text, BigramLanguageModel
from miniTransformer.preprocessing.sourcing.sourcing import (
    load_data,
    create_char_mappings,
)
from miniTransformer.visuzalization.visualize_attention import create_animation
import torch
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a miniTransformer model.")

    parser.add_argument(
        "--root_dir",
        type=str,
        default="Code/juan-garassino",  # Specify the path as a single string
    )
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
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/miniTransformer/data",
    )
    parser.add_argument("--name", type=str, default="input.txt")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/miniTransformer/results/checkpoints",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="/miniTransformer/results/heatmaps"
    )
    parser.add_argument(
        "--animation_dir", type=str, default="/miniTransformer/results/animation"
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.generate:
        if args.colab == 0:
            # Assuming args.data_dir has been set up with another parser.add_argument call
            args.data_dir = os.path.join(
                os.environ.get("HOME"), args.root_dir, args.data_dir
            )

            args.checkpoint_dir = os.path.join(
                os.environ.get("HOME"), args.root_dir, args.checkpoint_dir
            )
            
            checkpoint_dir = (args.checkpoint_dir,)

        if args.checkpoint:

            device = torch.device(args.device)
            print(checkpoint_dir)
            print(args.checkpoint)
            checkpoint = torch.load(
                os.path.join(checkpoint_dir[0], args.checkpoint), map_location=device
            )
            
            model_state_dict = checkpoint["model_state_dict"]

            char_to_int, int_to_char, vocab_size = create_char_mappings(
                load_data(args.data_dir)  # , args.name)
            )
            
            model = BigramLanguageModel(
                vocab_size,
                args.n_embd,
                args.block_size,
                args.n_head,
                args.n_layer,
                device,
            ).to(device)
            
            model.load_state_dict(model_state_dict)

            model.eval()

            print("\n")

            for char in generate_text(
                model, int_to_char, device, max_new_tokens=args.n_of_char
            ):
                print(char, end="", flush=True)
                sys.stdout.flush()

        else:
            print("Please provide a checkpoint file to generate text.")
    
    else:

        args.data_dir = os.path.join(
            os.environ.get("HOME"), args.root_dir, args.data_dir
        )

        args.checkpoint_dir = os.path.join(
            os.environ.get("HOME"), args.root_dir, args.checkpoint_dir
        )

        args.plots_dir = os.path.join(
            os.environ.get("HOME"), args.root_dir, args.plots_dir
        )

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
            checkpoint_dir=args.checkpoint_dir,
            plots_dir=args.plots_dir,
            animation_dir=args.animation_dir
        )
