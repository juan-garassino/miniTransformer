import argparse
import torch

def parse_arguments():
    """Parse command line arguments for the miniTransformer script."""
    parser = argparse.ArgumentParser(description="Train a miniTransformer model.")
    parser.add_argument("--root_dir", type=str, default="Code/juan-garassino")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--embd_dim", type=int, default=64)
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