import os
import torch
from colorama import Fore, Style
from miniTransformer.sourcing.sourcing import load_data, create_train_val_splits
from miniTransformer.preprocessing.tokenizers.simple_tokenizer import SimpleTokenizer
from miniTransformer.model.bigram_language_model import BigramLanguageModel
from miniTransformer.model.losses import estimate_loss, create_data_batch
from miniTransformer.evaluate.visualize_attention import (
    visualize_attention,
    create_animation,
)
import sys
from miniTransformer.preprocessing.tokenizers.regex_tokenizer import RegexTokenizer


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

    # print(f'\n')
    print(f"\n\n✅ {Fore.YELLOW}Saved checkpoint at step {epoch}{Style.RESET_ALL}")





def train(
    batch_size=16,
    block_size=32,
    vocab_size=256,
    max_iters=1000,
    tokenizer='regex',
    eval_interval=100,
    learning_rate=1e-3,
    device='cpu',
    eval_iters=10,
    embd_dim=32,
    n_head=4,
    n_layer=4,
    dropout=0.0,
    colab=0,
    path=None,
    name=None,
    save_interval=25,
    heatmaps_dir=25,
    animations_dir=None,
    checkpoints_dir=None,
    heatmap_interval=25,
):
    """
    Train the BigramLanguageModel.

    :param batch_size: Batch size for training
    :param block_size: Block size for the model
    :param max_iters: Maximum number of iterations for training
    :param eval_interval: Interval to evaluate the loss on train and val sets
    :param learning_rate: Learning rate for the optimizer
    :param device: Device (CPU or GPU) to run the model on
    :param eval_iters: Number of iterations for loss estimation
    :param embd_dim: Embedding size
    :param n_head: Number of attention heads
    :param n_layer: Number of layers
    :param dropout: Dropout rate
    :param colab: Flag for running on Google Colab
    :param path: Path to the text file
    :param name: Name of the dataset
    :param save_interval: Interval to save the model checkpoints
    :param checkpoints_dir: Directory to save the model checkpoints
    :param heatmap_interval: Interval to save attention heatmaps
    """
    print(f"\n✅ {Fore.CYAN}Loading the data...{Style.RESET_ALL}")
    text = load_data(path)  # , name)

    print(f"\n🔀 {Fore.CYAN}Creating character mappings using {tokenizer} tokenizer...{Style.RESET_ALL}")
    #char_to_int, int_to_char, vocab_size = create_char_mappings(text)
    regex_tokenizer = RegexTokenizer()

    project_root = os.environ.get("PROJECT_ROOT")

    results_file_path = os.path.join(project_root, "results", "tokenizers")

    if not os.path.exists(results_file_path):
        os.makedirs(results_file_path)

    name = f'{tokenizer}{vocab_size}'

    prefix = os.path.join(results_file_path, name)

    regex_tokenizer.train(text, vocab_size=vocab_size, verbose=True)

    regex_tokenizer.save(prefix)

    print(f"\n🔢 {Fore.CYAN}Creating encoder and decoder functions...{Style.RESET_ALL}")
    #encoder, decoder = create_encoder_decoder(char_to_int, int_to_char)
    encoded_text = regex_tokenizer.encode(text)

    print(f"\n🔤 {Fore.CYAN}Encoding the input text...{Style.RESET_ALL}")
    #encoded_text = encode_text(text)

    print(
        f"\n🔄 {Fore.CYAN}Creating training and validation data splits...{Style.RESET_ALL}"
    )
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    print(f"\n🔄 {Fore.CYAN}Instantiating the BigramLanguageModel...{Style.RESET_ALL}")

    model = BigramLanguageModel(vocab_size=vocab_size, embd_dim=embd_dim, block_size=block_size, n_head=n_head, n_layer=n_layer, dropout=dropout, device=device)

    m = model.to(device)
    print(f"\n🔄 {Fore.GREEN}Moved the model to the device{Style.RESET_ALL}")

    print(f"\n✅ {Fore.CYAN}Creating a PyTorch optimizer...{Style.RESET_ALL}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in m.parameters()) / 1e6

    print(
        f"\n✅ {Fore.MAGENTA}The total number of parameters is {total_params} million{Style.RESET_ALL}", end=f'\n\n'
    )

    for iter in range(max_iters):
        # Sample a batch of data
        print(f"\r✅ {Fore.CYAN}Sampling a batch of data...{Style.RESET_ALL}", end=f'')
        
        xb, yb = create_data_batch(
            train_data,
            val_data,
            "train",
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )
        
        # Evaluate the loss and update the model
        print(f"\r✅ {Fore.CYAN}Updating the model parameters @ {iter}...{Style.RESET_ALL}", end=f'')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save model periodically
        if iter % save_interval == 0 or iter == max_iters - 1:
            
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
                
                #print(f'\n')
                print(f"\n✅ {Fore.GREEN}Checkpoint directory was created{Style.RESET_ALL}")

            save_checkpoint(
                model,
                optimizer,
                iter,
                os.path.join(checkpoints_dir, f"checkpoint_{iter}.pt"),
            ) # TODO save checkpoint from the model class?

        # Evaluation periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"\n✅ {Fore.CYAN}Evaluating model loss...{Style.RESET_ALL}")

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                eval_iters,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
            )
            print(
                f"\n✅ {Fore.MAGENTA}Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{Style.RESET_ALL}", end=f'\n'
            )

        # Save attention heatmaps periodically
        if iter % heatmap_interval == 0 or iter == max_iters - 1:

            print(f"\n✅ {Fore.CYAN}Saving attention heatmaps...{Style.RESET_ALL}")

            # Assuming `model.attention_heads_matrices()` returns the structure [block, head, [Q, V, K]]
            attention_matrices = model.attention_heads

            # Number of layers (blocks)
            num_layers = len(attention_matrices)

            print(
                f"\n✅ {Fore.CYAN}The number of layers is {num_layers}...{Style.RESET_ALL}"
            )

            # Assuming all layers have the same number of heads
            num_heads_per_layer = len(attention_matrices[0]) if num_layers > 0 else 0

            print(
                f"\n✅ {Fore.CYAN}The number of heads per layer is {num_heads_per_layer}...{Style.RESET_ALL}"
            )

            # Prepare input tensors for visualization
            # Creating a structure for Q, K, V separately, each will be a list of tensors
            # where each tensor represents all heads across all layers for that type
            input_tensors_Q, input_tensors_K, input_tensors_V = [], [], []

            for matrix_type in range(3):  # 0 for Q, 1 for K, 2 for V
                # For each type, create a tensor that combines all layers and heads
                combined_matrix = [
                    torch.stack(
                        [
                            attention_matrices[layer][head][matrix_type]
                            for head in range(num_heads_per_layer)
                        ]
                    )
                    for layer in range(num_layers)
                ]

                # Append the combined matrix to the respective list
                if matrix_type == 0:
                    input_tensors_Q.append(torch.stack(combined_matrix))
                elif matrix_type == 1:
                    input_tensors_K.append(torch.stack(combined_matrix))
                else:
                    input_tensors_V.append(torch.stack(combined_matrix))

            print(
                f"\n✅ {Fore.CYAN}Shape of combined Q tensor: {input_tensors_Q[0].shape}{Style.RESET_ALL}"
            )

            print(
                f"\n✅ {Fore.CYAN}Shape of combined K tensor: {input_tensors_K[0].shape}{Style.RESET_ALL}"
            )

            print(
                f"\n✅ {Fore.CYAN}Shape of combined V tensor: {input_tensors_V[0].shape}{Style.RESET_ALL}"
            )

            # At this point, input_tensors_Q, input_tensors_K, and input_tensors_V
            # each contains a single tensor structured as [layer, head, ...matrix dimensions...]

            # Now, visualize each tensor type in a grid
            for tensors, name in zip(
                [input_tensors_Q, input_tensors_K, input_tensors_V],
                ["Keys", "Values", "Queries"],
            ):
                visualize_attention(
                    tensors,
                    name,
                    output_dir=heatmaps_dir,
                    animation_dir=animations_dir,
                    iter_num=iter,
                    animation=True,
                    grid_size=(
                        num_layers,
                        num_heads_per_layer,
                    ),  # Assuming visualize_attention supports grid_size parameter
                )

            print(
                f"\n✅ {Fore.YELLOW}Saved attention heatmaps at step {iter}{Style.RESET_ALL}"
            )

        # Save final animation
        if iter == max_iters - 1:
            tensor_names = ["K", "V", "Q"]
            # Call the function to create animations for all tensor types
            create_animation(
                tensor_names, output_dir=heatmaps_dir, animation_dir=animations_dir
            )
            print(
                f"\n🎦 {Fore.BLUE}All animations created after iteration {iter + 1}.{Style.RESET_ALL}"
            )


if __name__ == "__main__":
    # Set default hyperparameters and constants
    # Call the train function with the default values
    train(
        batch_size=16,
        block_size=32,
        max_iters=500,
        eval_interval=100,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_iters=200,
        embd_dim=64,
        n_head=4,
        n_layer=4,
        dropout=0.0,
        colab=1,
        path=os.path.join(
            os.environ.get("HOME"),
            "Users",
            "juan-garassino",
            "Code",
            "juan-garassino",
            "miniTransformer",
            "miniTransformer",
            "data",
        ),
        name="input.txt",
        save_interval=100,
        checkpoints_dir=os.path.join(
            os.environ.get("HOME"),
            "Users",
            "juan-garassino",
            "Code",
            "juan-garassino",
            "miniTransformer",
            "miniTransformer",
            "checkpoints",
        ),
    )
