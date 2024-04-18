import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from miniTransformer.model.transformer_block import Block

dropout = 0.0
block_size = 32  # what is the maximum context length for predictions?


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model using a Transformer architecture.

    Args:
        vocab_size (int): The size of the vocabulary.
        embd_dim (int): The size of the token and position embeddings.
        block_size (int): The sequence length (block size) of the input.
        n_head (int): The number of attention heads.
        n_layer (int): The number of Transformer layers.
        device (torch.device): The device to run the model on (CPU or GPU).
    """

    def __init__(
        self,
        vocab_size=256,
        embd_dim=64,
        block_size=32,
        n_head=4,
        n_layer=4,
        dropout=0.0,
        device=None,
    ):
        super().__init__()

        print(
            f"\n🔄 {Fore.CYAN}Instantiating the BigramLanguageModel...{Style.RESET_ALL}"
        )

        # Define the token and position embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding_table = nn.Embedding(block_size, embd_dim)

        # Create the Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(embd_dim, n_head, dropout) for _ in range(n_layer)]
        )

        # Define the final layer normalization layer
        self.ln_f = nn.LayerNorm(embd_dim)

        # Define the language model head
        self.lm_head = nn.Linear(embd_dim, vocab_size)

        self.device = device

        print(f"\n🔢 {Fore.YELLOW}Number of Attention Heads: {n_head}{Style.RESET_ALL}")

        print(f"\n🔢 {Fore.YELLOW}Embedding Size: {embd_dim}{Style.RESET_ALL}")

        print(f"\n🔢 {Fore.YELLOW}Block Size: {block_size}{Style.RESET_ALL}")

        print(f"\n✅ {Fore.CYAN}BigramLanguageModel Initialized...{Style.RESET_ALL}")

    def forward(self, idx, targets=None):
        """
        Forward pass through the BigramLanguageModel.

        Args:
            idx (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): The target tensor of shape (batch_size, sequence_length).

        Returns:
            logits (torch.Tensor): The logits tensor of shape (batch_size, sequence_length, vocab_size).
            loss (torch.Tensor): The loss tensor (scalar) if targets are provided, else None.
        """
        B, T = idx.shape

        # Get the token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)

        # Combine the token and position embeddings
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass the combined embeddings through the Transformer blocks
        for block in self.blocks:
            x, _ = block(x)  # (B, T, C)

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
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            yield idx_next

    @property
    def attention_heads(self):
        """
        A property that returns the matrices for Queries (Q), Values (V), and Keys (K)
        from all attention heads across all Transformer blocks. The structure is organized
        by block, then head, with each head containing a list [Q, V, K] matrices.

        Returns:
            all_attention_heads (list of list of lists): A nested list structure where
            the first dimension is the block, the second is the head within that block,
            and the third contains the Q, V, K matrices in order.
        """
        # Initialize an empty list to hold the structured attention heads data
        all_attention_heads = []

        # Iterate over each block in the model
        for block in self.blocks:
            # Initialize a list for the current block's heads
            block_heads = []

            # Iterate over each head in the current block
            for head in block.sa.heads:
                # Add the head's Q, V, K matrices to a list in specific order
                head_matrices = [
                    head.query.weight.data,  # Assuming .weight.data gives the actual matrix for Q
                    head.value.weight.data,  # For V
                    head.key.weight.data,  # For K
                ]
                # Append the head's matrices list to the current block's list of heads
                block_heads.append(head_matrices)

            # Append the current block's list of heads (with their matrices) to the all_attention_heads list
            all_attention_heads.append(block_heads)

        return all_attention_heads
