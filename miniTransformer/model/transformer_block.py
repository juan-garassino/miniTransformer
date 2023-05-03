import torch.nn as nn

from miniTransformer.model.head import MultiHeadAttention, FeedForward


class Block(nn.Module):
    """
    Transformer block: communication (self-attention) followed by computation (feed-forward).
    """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        # Calculate head_size by dividing the embedding dimension by the number of heads
        head_size = n_embd // n_head

        # Instantiate the multi-head self-attention layer
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)

        # Instantiate the feed-forward layer
        self.ffwd = FeedForward(n_embd, dropout)

        # Instantiate the layer normalization layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply the multi-head self-attention layer and add the residual connection
        x = x + self.sa(self.ln1(x))

        # Apply the feed-forward layer and add the residual connection
        x = x + self.ffwd(self.ln2(x))

        return x
