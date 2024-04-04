import os
import torch
from colorama import Fore, Style
from miniTransformer.preprocessing.sourcing.sourcing import (
    load_data,
    create_char_mappings,
    create_encoder_decoder,
    create_train_val_splits,
)
from miniTransformer.architecture.bigram import BigramLanguageModel
from miniTransformer.architecture.losses import estimate_loss, create_data_batch
from miniTransformer.visuzalization.visualize_attention import visualize_attention, create_animation
import sys


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

    print(f"\n✅ {Fore.YELLOW}Saved checkpoint at step {epoch}{Style.RESET_ALL}")


import sys


def generate_text(model, int_to_char, device, max_new_tokens=200):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_tokens = model.generate_iter(context, max_new_tokens=max_new_tokens)

    for tokens in generated_tokens:
        for token in tokens:
            token = token.item()
            if token in int_to_char:
                char = int_to_char[token]
                yield char


def train(
    batch_size,
    block_size,
    max_iters,
    eval_interval,
    learning_rate,
    device,
    eval_iters,
    n_embd,
    n_head,
    n_layer,
    dropout,
    colab,
    path,
    name,
    save_interval,
    heatmaps_dir,
    animations_dir,
    checkpoints_dir,
    heatmap_interval,
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
    :param n_embd: Embedding size
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

    print(f"\n🔀 {Fore.CYAN}Creating character mappings...{Style.RESET_ALL}")
    char_to_int, int_to_char, vocab_size = create_char_mappings(text)

    print(f"\n🔢 {Fore.CYAN}Creating encoder and decoder functions...{Style.RESET_ALL}")
    encode_text, decode_list = create_encoder_decoder(char_to_int, int_to_char)

    print(f"\n🔤 {Fore.CYAN}Encoding the input text...{Style.RESET_ALL}")
    encoded_text = encode_text(text)

    print(
        f"\n🔄 {Fore.CYAN}Creating training and validation data splits...{Style.RESET_ALL}"
    )
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    print(f"\n🔄 {Fore.CYAN}Instantiating the BigramLanguageModel...{Style.RESET_ALL}")
    
    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, device)

    m = model.to(device)
    print(f"\n🔄 {Fore.GREEN}Moved the model to the device{Style.RESET_ALL}")

    print(f"\n✅ {Fore.CYAN}Creating a PyTorch optimizer...{Style.RESET_ALL}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in m.parameters()) / 1e6

    print(f"\n✅ {Fore.MAGENTA}The total number of parameters is {total_params} million{Style.RESET_ALL}")

    for iter in range(max_iters):
        if iter % save_interval == 0 or iter == max_iters - 1:
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            print(f"\n✅ {Fore.GREEN}Checkpoint directory was created{Style.RESET_ALL}")

            save_checkpoint(
                model,
                optimizer,
                iter,
                os.path.join(checkpoints_dir, f"checkpoint_{iter}.pt"),
            )

            # HERE WAS THE EVALUATION INDENTED INSIDE HERE

            # Sample a batch of data
            print(f"\n✅ {Fore.CYAN}Sampling a batch of data...{Style.RESET_ALL}")

            xb, yb = create_data_batch(
                train_data,
                val_data,
                "train",
                block_size=block_size,
                batch_size=batch_size,
                device=device,
            )

            # Evaluate the loss and update the model
            print(f"\n✅ {Fore.CYAN}Updating the model parameters...{Style.RESET_ALL}")

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

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
                f"\n✅ {Fore.MAGENTA}step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{Style.RESET_ALL}"
            )

        # THIS WAS ALSO INDENTED INSIDE

        # Save attention heatmaps periodically
        if iter % heatmap_interval == 0 or iter == max_iters - 1:

            print(f"\n✅ {Fore.CYAN}Saving attention heatmaps...{Style.RESET_ALL}")

            # Assuming `model.attention_heads_matrices()` returns the structure [block, head, [Q, V, K]]
            attention_matrices = model.attention_heads

            # Number of layers (blocks)
            num_layers = len(attention_matrices)

            print(f"\n✅ {Fore.CYAN}The number of layers is {num_layers}...{Style.RESET_ALL}")

            # Assuming all layers have the same number of heads
            num_heads_per_layer = len(attention_matrices[0]) if num_layers > 0 else 0

            print(f"\n✅ {Fore.CYAN}The number of heads per layer is {num_heads_per_layer}...{Style.RESET_ALL}")

            # Prepare input tensors for visualization
            # Creating a structure for Q, K, V separately, each will be a list of tensors
            # where each tensor represents all heads across all layers for that type
            input_tensors_Q, input_tensors_K, input_tensors_V = [], [], []

            for matrix_type in range(3):  # 0 for Q, 1 for K, 2 for V
                # For each type, create a tensor that combines all layers and heads
                combined_matrix = [
                    torch.stack([attention_matrices[layer][head][matrix_type] for head in range(num_heads_per_layer)])
                    for layer in range(num_layers)
                ]
                
                # Append the combined matrix to the respective list
                if matrix_type == 0:
                    input_tensors_Q.append(torch.stack(combined_matrix))
                elif matrix_type == 1:
                    input_tensors_K.append(torch.stack(combined_matrix))
                else:
                    input_tensors_V.append(torch.stack(combined_matrix))

            print(f"\n✅ {Fore.CYAN}Shape of combined Q tensor: {input_tensors_Q[0].shape}{Style.RESET_ALL}")
                
            print(f"\n✅ {Fore.CYAN}Shape of combined K tensor: {input_tensors_K[0].shape}{Style.RESET_ALL}")
                
            print(f"\n✅ {Fore.CYAN}Shape of combined V tensor: {input_tensors_V[0].shape}{Style.RESET_ALL}")
            
            # At this point, input_tensors_Q, input_tensors_K, and input_tensors_V
            # each contains a single tensor structured as [layer, head, ...matrix dimensions...]

            # Now, visualize each tensor type in a grid
            for tensors, name in zip([input_tensors_Q, input_tensors_K, input_tensors_V], ["Keys", "Values", "Queries"]):
                visualize_attention(
                    tensors, name, output_dir=heatmaps_dir, animation_dir=animations_dir, iter_num=iter, animation=True,
                    grid_size=(num_layers, num_heads_per_layer)  # Assuming visualize_attention supports grid_size parameter
                )

            print(
                f"\n✅ {Fore.YELLOW}Saved attention heatmaps at step {iter}{Style.RESET_ALL}"
            )

        if iter == max_iters - 1:
            tensor_names = ["K", "V", "Q"]
            # Call the function to create animations for all tensor types
            create_animation(tensor_names, output_dir=heatmaps_dir, animation_dir=animations_dir)
            print(f"\n🎦 {Fore.BLUE}All animations created after iteration {iter + 1}.{Style.RESET_ALL}")

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
        n_embd=64,
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
