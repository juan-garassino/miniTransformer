import torch.nn as nn

from miniTransformer.model.multihead_attention import MultiHeadAttention, FeedForward


class Block(nn.Module):
    """
    Transformer block: communication (self-attention) followed by computation (feed-forward).
    """

    def __init__(self, embd_dim, n_head, dropout):
        # embd_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        # Calculate head_size by dividing the embedding dimension by the number of heads
        head_size = embd_dim // n_head

        # Instantiate the multi-head self-attention layer
        self.sa = MultiHeadAttention(n_head, head_size, embd_dim, dropout)

        # Instantiate the feed-forward layer
        self.ffwd = FeedForward(embd_dim, dropout)

        # Instantiate the layer normalization layers
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x_norm = self.ln1(x)
        x_att, attns = self.sa(x_norm)
        x = x + x_att

        # Apply the feed-forward layer and add the residual connection
        x = x + self.ffwd(self.ln2(x))

        # Return the output and the attention matrices, keys, queries, and values
        return x, attns
