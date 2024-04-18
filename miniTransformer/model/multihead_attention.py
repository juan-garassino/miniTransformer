import torch
from torch import nn
import torch.nn.functional as F
from colorama import Fore, Style


class AttentionHead(nn.Module):
    """
    One head of self-attention in a multi-head self-attention mechanism.
    """

    def __init__(self, head_size, embd_dim, block_size, dropout, verbose=False):
        super().__init__()

        # Define the key, query, and value linear layers without bias
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)
        self.verbose = verbose

        # Register a lower triangular matrix used for masking
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get the dimensions of the input tensor
        B, T, C = x.shape

        # print(Fore.YELLOW + f"\nX shape {x.shape}" + Style.RESET_ALL)

        # Calculate the keys, queries, and values using the linear layers
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute attention scores ("affinities") using matrix multiplication
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)

        # Apply the mask to the attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        # Normalize the attention scores using the softmax function
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Apply dropout to the attention scores
        wei = self.dropout(wei)

        # Calculate the values using the value linear layer
        v = self.value(x)  # (B, T, C)

        # Perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        if self.verbose:
            print(Fore.YELLOW + f"\nKeys shape {k.shape}" + Style.RESET_ALL)

            print(Fore.CYAN + f"\nQueries shape {q.shape}" + Style.RESET_ALL)

            print(Fore.MAGENTA + f"\nWeights shape {wei.shape}" + Style.RESET_ALL)

            print(Fore.GREEN + f"\nValues shape {v.shape}" + Style.RESET_ALL)

            print(Fore.BLUE + f"\nOutput shape {out.shape}" + Style.RESET_ALL)

        # Return the output, keys, queries, and values
        return out, k, q, v


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, num_heads, head_size, embd_dim, dropout):
        super().__init__()

        # Create a module list of self-attention heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, embd_dim, embd_dim, dropout)
                for _ in range(num_heads)
            ]
        )

        # Define the output projection linear layer
        self.proj = nn.Linear(embd_dim, embd_dim)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of each head along the last dimension
        out = torch.cat([h(x)[0] for h in self.heads], dim=-1)

        # Collect attention matrices, keys, queries, and values from each head
        attns = [h(x)[1:] for h in self.heads]

        # print([type(attn) for attn in attns])
        # print([attn.shape for attn in attns])

        # Apply the output projection and dropout
        out = self.dropout(self.proj(out))

        return out, attns


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, embd_dim, dropout):
        super().__init__()

        # Define the feed-forward network using a sequential container
        self.net = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.ReLU(),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pass the input through the feed-forward network
        return self.net(x)
