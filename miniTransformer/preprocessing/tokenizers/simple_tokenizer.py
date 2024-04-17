from miniTransformer.sourcing.sourcing import create_train_val_splits, load_data
from miniTransformer.utils.parse_arguments import parse_arguments
from miniTransformer.preprocessing.tokenizers.base_tokenizer import Tokenizer

import os
from colorama import Fore, Style

class SimpleTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text):
        """
        Create mappings between characters and integers for the given text.

        Args:
            text (str): The input text.

        Returns:
            char_to_int (dict): A dictionary mapping characters to integers.
            int_to_char (dict): A dictionary mapping integers to characters.
            vocab_size (int): The number of unique characters in the text.
        """
        # Get all the unique characters in the text
        unique_chars = sorted(list(set(text)))
        vocab_size = len(unique_chars)

        # Create mappings between characters and integers
        char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        int_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        # Print the length of the dataset in characters
        print(f"\n✅ {Fore.MAGENTA}Length of dataset in characters: {len(text)}{Style.RESET_ALL}")

        # Print the mappings and vocab size
        print(f"\n✅ {Fore.MAGENTA}Character to Integer Mapping: {list(char_to_int.items())[:3]}{Style.RESET_ALL}")
        print(f"\n✅ {Fore.MAGENTA}Integer to Character Mapping: {list(int_to_char.items())[:3]}{Style.RESET_ALL}")
        print(f"\n✅ {Fore.MAGENTA}Vocabulary Size: {vocab_size}{Style.RESET_ALL}")

        return char_to_int, int_to_char, vocab_size


    def encode_text(self, char_to_int):
        """
        Create an encoding function for text data.

        Args:
            char_to_int (dict): A dictionary mapping characters to integers.

        Returns:
            encode_text (callable): A function that encodes a string to a list of integers.
        """
        # Encoder: convert a string to a list of integers
        def encode_text(text):
            return [char_to_int[c] for c in text]

        return encode_text

    def decode_text(self, int_to_char):
        """
        Create a decoding function for text data.

        Args:
            int_to_char (dict): A dictionary mapping integers to characters.

        Returns:
            decode_list (callable): A function that decodes a list of integers to a string.
        """
        # Decoder: convert a list of integers to a string
        def decode(l):
            return "".join([int_to_char[i] for i in l])

        return decode

if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join(os.environ.get("HOME"), args.root_dir, args.data_dir.lstrip("/"))

    data = load_data(path)

    simple_tokenizer = SimpleTokenizer()

    # Create character to integer and integer to character mappings
    char_to_int, int_to_char, vocab_size = simple_tokenizer.train(data)

    # Create encoder and decoder functions
    #encoder, decoder = simple_tokenizer.train(char_to_int, int_to_char)

    # Encode the input text
    encoder = simple_tokenizer.encode_text(char_to_int)

    decoder = simple_tokenizer.decode_text(int_to_char)

    encoded_text = encoder(data)

    # Create training and validation data splits
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    # Display the training and validation data
    print("Train data:", train_data[:10])

    print("Validation data:", val_data[:10])

    # Display the encoded and decoded text
    print("Encoded text:", encoded_text[:10])
    decoded_text = decoder(encoded_text)
    print("Decoded text:", decoded_text[:10])