import torch
import os

from miniTransformer.model.bigram import BigramLanguageModel
from miniTransformer.model.losses import estimate_loss
from miniTransformer.sourcing.sourcing import create_data_batch, create_char_mappings, create_encoder_decoder, create_train_val_splits, load_data

batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

path = '/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/'

name = 'input.txt'

text = load_data(path, name)

# Create character to integer and integer to character mappings
char_to_int, int_to_char, vocab_size = create_char_mappings(text)

# Create encoder and decoder functions
encode_text, decode_list = create_encoder_decoder(char_to_int, int_to_char)

# Encode the input text
encoded_text = encode_text(text)

# Create training and validation data splits
train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, device)

m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model,
                               train_data,
                               val_data,
                               eval_iters,
                               block_size=block_size,
                               batch_size=batch_size,
                               device=device)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = create_data_batch(train_data,
                               val_data,
                               'train',
                               block_size=block_size,
                               batch_size=batch_size,
                               device=device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode_list(m.generate(context, max_new_tokens=2000)[0].tolist()))
