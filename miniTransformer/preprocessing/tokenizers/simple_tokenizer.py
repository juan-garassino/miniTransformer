import os
from colorama import Fore, Style

from miniTransformer.sourcing.sourcing import create_train_val_splits, load_data
from miniTransformer.utils.parse_arguments import parse_arguments


class SimpleTokenizer:
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

        print(
            f"\n🔀 {Fore.CYAN}Creating character mappings using simple tokenizer...{Style.RESET_ALL}"
        )

        if vocab_size is None:
            unique_chars = sorted(list(set(text)))
            self.vocab_size = len(unique_chars)
        else:
            self.vocab_size = vocab_size

        # Create mappings between characters and integers
        self.char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        # Print verbose information if required
        if verbose:
            print(
                f"\n✅ {Fore.MAGENTA}Length of dataset in characters: {len(text)}{Style.RESET_ALL}"
            )
            print(
                f"\n✅ {Fore.MAGENTA}Character to Integer Mapping: {list(self.char_to_int.items())[:3]}{Style.RESET_ALL}"
            )
            print(
                f"\n✅ {Fore.MAGENTA}Integer to Character Mapping: {list(self.int_to_char.items())[:3]}{Style.RESET_ALL}"
            )
            print(
                f"\n✅ {Fore.MAGENTA}Vocabulary Size: {self.vocab_size}{Style.RESET_ALL}"
            )

        return self.vocab_size

    def encode(self, text):
        """
        Encode a string into a list of integers.

        Args:
            text (str): The input text.

        Returns:
            list: A list of integers representing the encoded text.
        """

        print(f"\n🔢 {Fore.CYAN}Creating encoder functions...{Style.RESET_ALL}")

        print(f"\n🔤 {Fore.CYAN}Encoding the input text...{Style.RESET_ALL}")

        return [self.char_to_int[c] for c in text]

    def decode(self, ids):
        """
        Decode a list of integers into a string.

        Args:
            ids (list): A list of integers representing the encoded text.

        Returns:
            str: The decoded text.
        """

        # print(f"\n🔢 {Fore.CYAN}Creating decoder functions...{Style.RESET_ALL}")

        # print(f"\n🔤 {Fore.CYAN}Decoding the input tokens...{Style.RESET_ALL}")

        return "".join([self.int_to_char[i] for i in ids])

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model.

        Args:
            file_prefix (str): The prefix for the files.

        Returns:
            None
        """

        print(f"\n🔢 {Fore.CYAN}Saving mapping dictionaty...{Style.RESET_ALL}")

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
                # Convert newline character to its escape sequence
                if char == "\n":
                    char = "<NEWLINE>"
                if char == " ":
                    char = "<SPACE>"
                f.write(f"[{char}] {idx}\n")

        # # Save dictionary using pickle
        # with open("my_dict.pickle", "wb") as f:
        #     pickle.dump(self.int_to_char, f)

        print(f"\n🔢 {Fore.CYAN}Saved mapping dictionaty...{Style.RESET_ALL}")

    def load(self, model_file):
        """
        Load the tokenizer from a model file.

        Args:
            model_file (str): The path to the model file.

        Returns:
            None
        """
        print(f"\n🔢 {Fore.CYAN}Loading mapping dictionaty...{Style.RESET_ALL}")

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
                # Split the line into character and index
                parts = line.strip().split(" ")
                print(parts)
                char = parts[0][1:-1]  # Extract character, removing square brackets
                if (
                    char == "<NEWLINE>"
                ):  # Check if the character is a newline escape sequence
                    char = "\n"  # Replace escape sequence with newline character
                if (
                    char == "<SPACE>"
                ):  # Check if the character is a newline escape sequence
                    char = " "  # Replace escape sequence with newline character
                idx = int(parts[1])  # Extract index
                char_to_int[char] = idx
                int_to_char[idx] = char

        self.pattern = pattern
        self.special_tokens = special_tokens
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char

        self.vocab_size = len(self.int_to_char)

        # with open("my_dict.pickle", "rb") as f:
        #     self.int_to_char = pickle.load(f)

        print(f"\n🔢 {Fore.CYAN}Loaded mapping dictionaty...{Style.RESET_ALL}")

        return self.vocab_size


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.data_dir.lstrip("/")
    )

    data = load_data(path)

    data = "\n\nhello\nmy name is juan"

    combine_tokenizer = SimpleTokenizer()

    # Create character to integer and integer to character mappings
    combine_tokenizer.train(data)

    # Create encoder and decoder functions
    # encoder, decoder = simple_tokenizer.train(char_to_int, int_to_char)

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

    path = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.tokenizers_dir.lstrip("/"), "simple"
    )

    combine_tokenizer.save(path)

    path = os.path.join(
        os.environ.get("HOME"),
        args.root_dir,
        args.tokenizers_dir.lstrip("/"),
        "simple.model",
    )

    combine_tokenizer_loaded = SimpleTokenizer()

    combine_tokenizer_loaded.load(path)

    encoded_text = combine_tokenizer_loaded.encode(data)

    print(encoded_text)

    decoded_text = combine_tokenizer_loaded.decode(encoded_text)

    # Display the encoded and decoded text
    print("Encoded text:", encoded_text[:])

    print("Decoded text:", decoded_text[:])
