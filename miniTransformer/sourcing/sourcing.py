import torch

def load_data(path, name):
    """
    Load the text data from a file and return its content.

    Args:
        path (str): The path to the directory containing the text file.

    Returns:
        text (str): The content of the text file.
    """
    # Read the text file and store its content in a variable
    with open(f'{path}/{name}', 'r', encoding='utf-8') as f:
        text = f.read()

    print("Length of dataset in characters:", len(text))

    return text


def create_char_mappings(text):
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

    return char_to_int, int_to_char, vocab_size


def create_encoder_decoder(char_to_int, int_to_char):
    """
    Create encoding and decoding functions for text data.

    Args:
        char_to_int (dict): A dictionary mapping characters to integers.
        int_to_char (dict): A dictionary mapping integers to characters.

    Returns:
        encode_text (callable): A function that encodes a string to a list of integers.
        decode_list (callable): A function that decodes a list of integers to a string.
    """
    # Encoder: convert a string to a list of integers
    encode_text = lambda s: [char_to_int[c] for c in s]

    # Decoder: convert a list of integers to a string
    decode_list = lambda l: ''.join([int_to_char[i] for i in l])

    return encode_text, decode_list


def create_train_val_splits(encoded_text, train_ratio=0.9):
    """
    Create training and validation data splits from the encoded text data.

    Args:
        encoded_text (list): The encoded text data as a list of integers.
        train_ratio (float): The proportion of data to use for training (default: 0.9).

    Returns:
        train_data (torch.Tensor): A tensor containing the training data.
        val_data (torch.Tensor): A tensor containing the validation data.
    """
    # Convert the encoded text to a PyTorch tensor
    data = torch.tensor(encoded_text, dtype=torch.long)

    # Calculate the index where the training data ends and validation data begins
    n = int(train_ratio * len(data))

    # Split the data into training and validation sets
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def create_data_batch(train_data, val_data, split,
                      block_size=None,
                      batch_size=None,
                      device=None):
    """
    Generate a small batch of data consisting of inputs and targets.

    Args:
        split (str): The data split to use ('train' or 'val').
        block_size (int): The length of each data sequence.
        batch_size (int): The number of data sequences in the batch.
        device (torch.device): The device to use for tensor calculations.

    Returns:
        input_batch (torch.Tensor): A tensor containing the input data sequences.
        target_batch (torch.Tensor): A tensor containing the target data sequences.
    """
    data = train_data if split == 'train' else val_data

    indices = torch.randint(len(data) - block_size, (batch_size, ))

    input_batch = torch.stack([data[i:i + block_size] for i in indices])

    target_batch = torch.stack(
        [data[i + 1:i + block_size + 1] for i in indices])

    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    return input_batch, target_batch



if __name__ == "__main__":

    text = "Hello, World!"

    # Create character to integer and integer to character mappings
    char_to_int, int_to_char, vocab_size = create_char_mappings(text)

    # Create encoder and decoder functions
    encode_text, decode_list = create_encoder_decoder(char_to_int, int_to_char)

    # Encode the input text
    encoded_text = encode_text(text)

    # Create training and validation data splits
    train_data, val_data = create_train_val_splits(encoded_text,
                                                   train_ratio=0.9)

    # Display the training and validation data
    print("Train data:", train_data)
    print("Validation data:", val_data)

    # Display the encoded and decoded text
    print("Encoded text:", encoded_text)
    decoded_text = decode_list(encoded_text)
    print("Decoded text:", decoded_text)
