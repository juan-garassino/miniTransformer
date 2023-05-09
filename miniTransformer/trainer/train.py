import os
import torch
from miniTransformer.sourcing.sourcing import (
    load_data,
    create_char_mappings,
    create_encoder_decoder,
    create_train_val_splits,
)
from miniTransformer.model.bigram import BigramLanguageModel
from miniTransformer.model.losses import estimate_loss, create_data_batch
from miniTransformer.visuzalization.visualize_attention import visualize_attention


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def generate_text(model, int_to_char, device, max_new_tokens=20000):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_tokens = model.generate_iter(context, max_new_tokens=max_new_tokens)

    for token in generated_tokens:
        char = int_to_char[token.item()]
        print(char, end="", flush=True)


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
    # Load the data from the file
    text = load_data(path, name)

    # Create character to integer and integer to character mappings
    char_to_int, int_to_char, vocab_size = create_char_mappings(text)

    # Create encoder and decoder functions
    encode_text, decode_list = create_encoder_decoder(char_to_int, int_to_char)

    # Encode the input text
    encoded_text = encode_text(text)

    # Create training and validation data splits
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    # Instantiate the BigramLanguageModel
    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, device)

    # Move the model to the device (CPU or GPU)
    m = model.to(device)

    # Print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Main training loop
    for iter in range(max_iters):
        # Save checkpoint periodically
        if iter % save_interval == 0 or iter == max_iters - 1:
            save_checkpoint(
                model,
                optimizer,
                iter,
                os.path.join(checkpoint_dir, f"checkpoint_{iter}.pt"),
            )

        # Evaluate the loss on train and val sets periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
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
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # Sample a batch of data
        xb, yb = create_data_batch(
            train_data,
            val_data,
            "train",
            block_size=block_size,
            batch_size=batch_size,
            device=device,
        )

        # Evaluate the loss and update the model
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save attention heatmaps periodically
        if iter % heatmap_interval == 0 or iter == max_iters - 1:
            input_tensors = [
                [head.key.weight for head in model.attention_heads],
                [head.value.weight for head in model.attention_heads],
                [head.query.weight for head in model.attention_heads],
            ]
            tensor_names = ["Keys", "Values", "Queries"]
            visualize_attention(input_tensors,
                                tensor_names,
                                num_heatmaps=n_head,
                                iter_num=iter)



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
        path="/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/",
        name="input.txt",
        save_interval=100,
        checkpoint_dir="/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/checkpoints",
    )
