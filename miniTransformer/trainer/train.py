import os
import torch
from colorama import Fore, Style
from miniTransformer.sourcing.sourcing import (
    load_data,
    create_char_mappings,
    create_encoder_decoder,
    create_train_val_splits,
)
from miniTransformer.model.bigram import BigramLanguageModel
from miniTransformer.model.losses import estimate_loss, create_data_batch
from miniTransformer.visuzalization.visualize_attention import visualize_attention
import sys

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

    print(
        f"\n✅ {Fore.YELLOW}Saved checkpoint at step {epoch}{Style.RESET_ALL}")


import sys


def generate_text(model, int_to_char, device, max_new_tokens=200):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_tokens = model.generate_iter(context,
                                           max_new_tokens=max_new_tokens)

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
    checkpoint_dir,
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
    :param checkpoint_dir: Directory to save the model checkpoints
    :param heatmap_interval: Interval to save attention heatmaps
    """
    print(f"\n✅ {Fore.CYAN}Loading the data...{Style.RESET_ALL}")
    text = load_data(path, name)

    print(f"\n🔀 {Fore.CYAN}Creating character mappings...{Style.RESET_ALL}")
    char_to_int, int_to_char, vocab_size = create_char_mappings(text)

    print(
        f"\n🔢 {Fore.CYAN}Creating encoder and decoder functions...{Style.RESET_ALL}"
    )
    encode_text, decode_list = create_encoder_decoder(char_to_int, int_to_char)

    print(f"\n🔤 {Fore.CYAN}Encoding the input text...{Style.RESET_ALL}")
    encoded_text = encode_text(text)

    print(
        f"\n🔄 {Fore.CYAN}Creating training and validation data splits...{Style.RESET_ALL}"
    )
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    print(
        f"\n🔄 {Fore.CYAN}Instantiating the BigramLanguageModel...{Style.RESET_ALL}"
    )
    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, device)

    m = model.to(device)
    print(f"\n🔄 {Fore.GREEN}Moved the model to the device{Style.RESET_ALL}")

    print(f"\n✅ {Fore.CYAN}Creating a PyTorch optimizer...{Style.RESET_ALL}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(
        f"\n✅ {Fore.CYAN}Starting the main training loop...{Style.RESET_ALL}")

    for iter in range(max_iters):

        if iter % save_interval == 0 or iter == max_iters - 1:

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            print(
                f"\n✅ {Fore.GREEN}Checkpoint directory was created{Style.RESET_ALL}"
            )

            save_checkpoint(
                model,
                optimizer,
                iter,
                os.path.join(checkpoint_dir, f"checkpoint_{iter}.pt"),
            )

            # HERE WAS THE EVALUATION INDENTED INSIDE HERE

            # Sample a batch of data
            print(
                f"\n✅ {Fore.CYAN}Sampling a batch of data...{Style.RESET_ALL}")

            xb, yb = create_data_batch(
                train_data,
                val_data,
                "train",
                block_size=block_size,
                batch_size=batch_size,
                device=device,
            )

            # Evaluate the loss and update the model
            print(
                f"\n✅ {Fore.CYAN}Updating the model parameters...{Style.RESET_ALL}"
            )

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if iter % eval_interval == 0 or iter == max_iters - 1:

            print(
                f"\n✅ {Fore.CYAN}Evaluating model loss...{Style.RESET_ALL}"
            )

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
            print(
                f"\n✅ {Fore.CYAN}Saving attention heatmaps...{Style.RESET_ALL}"
            )
            input_tensors = [
                [head.key.weight for head in model.attention_heads],
                [head.value.weight for head in model.attention_heads],
                [head.query.weight for head in model.attention_heads],
            ]
            tensor_names = ["Keys", "Values", "Queries"]

            visualize_attention(input_tensors, tensor_names, iter_num=iter)

            print(
                f"\n✅ {Fore.YELLOW}Saved attention heatmaps at step {iter}{Style.RESET_ALL}"
            )

# "/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/"
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
        path=os.path.join(os.environ.get('HOME'), "Users", "juan-garassino", "Code", "juan-garassino", "miniTransformer", "miniTransformer", "data"),
        name="input.txt",
        save_interval=100,
        checkpoint_dir=os.path.join(os.environ.get('HOME'), "Users", "juan-garassino", "Code", "juan-garassino", "miniTransformer", "miniTransformer", "checkpoints")
    )
