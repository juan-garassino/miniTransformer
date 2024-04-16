"""Main module for the miniTransformer package execution.

This script allows for training the miniTransformer model or generating text from a trained model.
"""

import os
from miniTransformer.utils.parse_arguments import parse_arguments
from miniTransformer.generate.generate import generate_text_from_checkpoint
from miniTransformer.training.train import train, generate_text, BigramLanguageModel

# Assuming create_animation is used elsewhere or will be used in future updates.

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
            generate_text_from_checkpoint(
                                        args.checkpoint_path,
                                        args.checkpoints_dir,
                                        args.data_dir,
                                        args.device,
                                        args.n_embd,
                                        args.block_size,
                                        args.n_head,
                                        args.n_layer,
                                        args.dropout,
                                        args.n_of_char
                                    )
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
