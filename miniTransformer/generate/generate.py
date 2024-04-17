import os
import sys
import torch

from miniTransformer.model.bigram_language_model import BigramLanguageModel
from miniTransformer.sourcing.sourcing import load_data
from miniTransformer.preprocessing.tokenizers.simple_tokenizer import SimpleTokenizer
from miniTransformer.preprocessing.tokenizers.regex_tokenizer import RegexTokenizer

def generate_text(model, int_to_char, device, max_new_tokens=200):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_tokens = model.generate_iter(context, max_new_tokens=max_new_tokens)

    for tokens in generated_tokens:
        for token in tokens:
            token = token.item()
            if token in int_to_char:
                char = int_to_char[token]
                yield char

def generate_text_from_checkpoint(
    checkpoint=None,
    checkpoints_dir=None,
    data_dir=None,
    device='cpu',
    embd_dim=64,
    block_size=32,
    n_head=4,
    n_layer=4,
    dropout=0.0,
    n_of_char=None,
    vocab_size=256
):
    """
    Generate text using a pre-trained model checkpoint.

    Args:
        checkpoint (str): Path to the model checkpoint file.
        checkpoints_dir (str): Directory containing model checkpoints.
        data_dir (str): Directory containing data for creating character mappings.
        device (str): Device to use for model inference (e.g., 'cpu', 'cuda').
        embd_dim (int): Embedding dimension for the model.
        block_size (int): Block size for the model.
        n_head (int): Number of attention heads in the model.
        n_layer (int): Number of layers in the model.
        dropout (float): Dropout rate for the model.
        n_of_char (int): Maximum number of characters to generate.

    Returns:
        None
    """
    if checkpoint:
        device = torch.device(device)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        data = load_data(data_dir)
        
        simple_tokenizer = SimpleTokenizer()
        
        regex_tokenizer = RegexTokenizer()
        
        _, int_to_char, _ = simple_tokenizer.train(data) # TODO here i need to load the tokenizer
        
        # tokenizer_model = '/Users/juan-garassino/Code/juan-garassino/miniNetworks/miniTransformer/results/tokenizers/regex260.model'
        
        # int_to_char = regex_tokenizer.load(tokenizer_model)
        
        model = BigramLanguageModel(
            vocab_size=vocab_size,
            embd_dim=embd_dim,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=device
        ).to(device)

        model.load_state_dict(model_state_dict)
        model.eval()
        print("\nGenerating text:\n")

        for char in generate_text(
            model, int_to_char, device, max_new_tokens=n_of_char
        ):
            print(char, end="", flush=True)
    else:
        print("Please provide a checkpoint file to generate text.", file=sys.stderr)
