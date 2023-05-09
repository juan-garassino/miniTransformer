import torch
import os

from miniTransformer.model.bigram import BigramLanguageModel
from miniTransformer.model.losses import estimate_loss
from miniTransformer.sourcing.sourcing import (
    create_data_batch,
    create_char_mappings,
    create_encoder_decoder,
    create_train_val_splits,
    load_data,
)

# Define hyperparameters and other constants
batch_size = 16
block_size = 32
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
colab=1

# Define the data file path
if colab == 1:
    path = "/content/miniTransformer/miniTransformer/data/"
else:
    path = "/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/"

name = "input.txt"

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

# Generate text from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated_tokens = m.generate_iter(context, max_new_tokens=20000)

for token in generated_tokens:
    char = int_to_char[token.item()]
    print(char, end="", flush=True)
