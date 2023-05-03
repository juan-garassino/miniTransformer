import torch
import torch.nn as nn
import torch.nn.functional as F

from miniTransformer.model.transformer_block import Block

dropout = 0.0
block_size = 32  # what is the maximum context length for predictions?


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model using a Transformer architecture.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device):
        super().__init__()

        # Define the token and position embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Create the Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        # Define the final layer normalization layer
        self.ln_f = nn.LayerNorm(n_embd)

        # Define the language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get the token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)

        # Combine the token and position embeddings
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass the combined embeddings through the Transformer blocks
        x = self.blocks(x)  # (B, T, C)

        # Apply the final layer normalization
        x = self.ln_f(x)  # (B, T, C)

        # Generate logits for each token in the vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate the loss if targets are provided
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate_iter(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
