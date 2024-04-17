from miniTransformer.sourcing.sourcing import create_train_val_splits, load_data
from miniTransformer.utils.parse_arguments import parse_arguments
from miniTransformer.preprocessing.tokenizers.base_tokenizer import Tokenizer
from miniTransformer.preprocessing.tokenizers.helpers import render_token

import os
from colorama import Fore, Style

import re

class CombinedTokenizer:
    """Combined class for Tokenizers"""

    def __init__(self):
        # Default values for attributes
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'': 100257}
        self.vocab = {}  # int -> bytes
        self.char_to_int = {}  # dict mapping characters to integers
        self.int_to_char = {}  # dict mapping integers to characters
        self.vocab_size = 0  # int

    def train(self, text, vocab_size=None, verbose=False):
        """
        Train the tokenizer by creating mappings between characters and integers for the given text.

        Args:
            text (str): The input text.
            vocab_size (int, optional): The number of unique characters in the text. If None, automatically calculated.
            verbose (bool, optional): Whether to print verbose information during training.

        Returns:
            None
        """
        if vocab_size is None:
            unique_chars = sorted(list(set(text)))
            self.vocab_size = len(unique_chars)
        else:
            self.vocab_size = vocab_size

        # Create mappings between characters and integers
        self.char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        print(self.char_to_int)
        print(self.int_to_char)

        # Print verbose information if required
        if verbose:
            print(f"\n✅ {Fore.MAGENTA}Length of dataset in characters: {len(text)}{Style.RESET_ALL}")
            print(f"\n✅ {Fore.MAGENTA}Character to Integer Mapping: {list(self.char_to_int.items())[:3]}{Style.RESET_ALL}")
            print(f"\n✅ {Fore.MAGENTA}Integer to Character Mapping: {list(self.int_to_char.items())[:3]}{Style.RESET_ALL}")
            print(f"\n✅ {Fore.MAGENTA}Vocabulary Size: {self.vocab_size}{Style.RESET_ALL}")

    def encode(self, text):
        """
        Encode a string into a list of integers.

        Args:
            text (str): The input text.

        Returns:
            list: A list of integers representing the encoded text.
        """
        return [self.char_to_int[c] for c in text]

    def decode(self, ids):
        """
        Decode a list of integers into a string.

        Args:
            ids (list): A list of integers representing the encoded text.

        Returns:
            str: The decoded text.
        """
        return "".join([self.int_to_char[i] for i in ids])

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model.

        Args:
            file_prefix (str): The prefix for the files.

        Returns:
            None
        """
        # Write to the model file
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

        # Write char_to_int and int_to_char to vocab file
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for char, idx in self.char_to_int.items():
                f.write(f"[{char}] {idx}\n")

    def load(self, model_file):
        """
        Load the tokenizer from a model file.

        Args:
            model_file (str): The path to the model file.

        Returns:
            None
        """
        assert model_file.endswith(".model")
        # Read the model file
        special_tokens = {}
        idx = 0  # Start index from 0

        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

        # Build char_to_int and int_to_char
        char_to_int = {}
        int_to_char = {}
        vocab_file = model_file.replace(".model", ".vocab")
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                # Split at the last occurrence of whitespace
                parts = line.rsplit(" ", 1)
                char = parts[0].strip()  # Remove leading and trailing whitespace
                if char.startswith("[") and char.endswith("]"):
                    char = char[1:-1]  # Remove square brackets
                char_idx = parts[1]  # Extract index
                char_to_int[char] = int(char_idx)
                int_to_char[int(char_idx)] = char

        self.pattern = pattern
        self.special_tokens = special_tokens
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char


        print(self.char_to_int)
        print(self.int_to_char)

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

    data = load_data(path).replace('\n', '')

    combine_tokenizer = CombinedTokenizer()

    # Create character to integer and integer to character mappings
    combine_tokenizer.train(data)

    # Create encoder and decoder functions
    #encoder, decoder = simple_tokenizer.train(char_to_int, int_to_char)

    # Encode the input text
    encoded_text = combine_tokenizer.encode(data)

    decoded_text = combine_tokenizer.decode(encoded_text)

    # Create training and validation data splits
    train_data, val_data = create_train_val_splits(encoded_text, train_ratio=0.9)

    # Display the training and validation data
    print("Train data:", train_data[:10])

    print("Validation data:", val_data[:10])

    # Display the encoded and decoded text
    print("Encoded text:", encoded_text[:10])

    print("Decoded text:", decoded_text[:10])

    path = os.path.join(os.environ.get("HOME"), args.root_dir, args.tokenizers_dir.lstrip("/"), 'combined')

    combine_tokenizer.save(path)
    
    path = os.path.join(os.environ.get("HOME"), args.root_dir, args.tokenizers_dir.lstrip("/"), 'combined.model')

    combine_tokenizer_loaded = CombinedTokenizer()

    combine_tokenizer_loaded.load(path)

    encoded_text = combine_tokenizer_loaded.encode(data)

    decoded_text = combine_tokenizer_loaded.decode(encoded_text)

    # Display the encoded and decoded text
    print("Encoded text:", encoded_text[-10:])

    print("Decoded text:", decoded_text[-10:])