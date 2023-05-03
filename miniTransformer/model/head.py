import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    One head of self-attention in a multi-head self-attention mechanism.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()

        # Define the key, query, and value linear layers without bias
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Register a lower triangular matrix used for masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get the dimensions of the input tensor
        B, T, C = x.shape

        # Calculate the keys, queries, and values using the linear layers
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute attention scores ("affinities") using matrix multiplication
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)

        # Apply the mask to the attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0,
                              float('-inf'))  # (B, T, T)

        # Normalize the attention scores using the softmax function
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Apply dropout to the attention scores
        wei = self.dropout(wei)

        # Calculate the values using the value linear layer
        v = self.value(x)  # (B, T, C)

        # Perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        # Return the output, keys, queries, and values
        return out, k, q, v


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()

        # Create a module list of self-attention heads
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, n_embd, dropout) for _ in range(num_heads)
        ])

        # Define the output projection linear layer
        self.proj = nn.Linear(n_embd, n_embd)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of each head along the last dimension
        out = torch.cat([h(x)[0] for h in self.heads], dim=-1)

        # Apply the output projection and dropout
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, n_embd, dropout):
        super().__init__()

        # Define the feed-forward network using a sequential container
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pass the input through the feed-forward network
        return self.net(x)
