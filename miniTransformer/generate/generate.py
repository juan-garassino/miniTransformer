import os
import sys
import torch

from miniTransformer.model.bigram_language_model import BigramLanguageModel
from miniTransformer.sourcing.sourcing import load_data
from miniTransformer.preprocessing.tokenizers.simple_tokenizer import SimpleTokenizer

def generate_text_from_checkpoint(
    checkpoint_path,
    checkpoints_dir,
    data_dir,
    device,
    n_embd,
    block_size,
    n_head,
    n_layer,
    dropout,
    n_of_char
):
    """
    Generate text using a pre-trained model checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        checkpoints_dir (str): Directory containing model checkpoints.
        data_dir (str): Directory containing data for creating character mappings.
        device (str): Device to use for model inference (e.g., 'cpu', 'cuda').
        n_embd (int): Embedding dimension for the model.
        block_size (int): Block size for the model.
        n_head (int): Number of attention heads in the model.
        n_layer (int): Number of layers in the model.
        dropout (float): Dropout rate for the model.
        n_of_char (int): Maximum number of characters to generate.

    Returns:
        None
    """
    if checkpoint_path:
        device = torch.device(device)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        data = load_data(data_dir)
        simple_tokenizer = SimpleTokenizer()
        char_to_int, int_to_char, vocab_size = simple_tokenizer.train(data) # TODO here i need to load the tokenizer
        
        model = BigramLanguageModel(
            vocab_size,
            n_embd,
            block_size,
            n_head,
            n_layer,
            dropout,
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
