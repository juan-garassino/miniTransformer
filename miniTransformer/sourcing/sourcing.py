import torch
import os
from colorama import Fore, Style

from miniTransformer.utils.parse_arguments import parse_arguments


def load_data(path):
    """
    Load the text data from all text files in a folder and return their concatenated content.

    Args:
        path (str): The path to the directory containing the text files.

    Returns:
        text (str): The concatenated content of all text files in the folder.
    """

    print(f"\n✅ {Fore.CYAN}Loading the data...{Style.RESET_ALL}")

    # Get a list of all the files in the directory
    files = [file for file in os.listdir(path) if file.endswith(".txt")]

    print(
        f"\n✅ {Fore.MAGENTA}There is {len(files)} files in the directory{Style.RESET_ALL}"
    )

    # Initialize an empty string to store the concatenated content
    text = ""

    # Iterate over each file in the directory
    for file_name in files:
        # Check if the file is a text file
        if file_name.endswith(".txt"):
            # Read the text file and append its content to the text string
            with open(os.path.join(path, file_name), "r", encoding="utf-8") as file:
                text += file.read()

    print(
        f"\n✅ {Fore.MAGENTA}Length of dataset in characters: {len(text)}{Style.RESET_ALL}"
    )

    return text


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

    print(
        f"\n🔄 {Fore.CYAN}Creating training and validation data splits...{Style.RESET_ALL}"
    )

    # Convert the encoded text to a PyTorch tensor
    data = torch.tensor(encoded_text, dtype=torch.long)

    # Calculate the index where the training data ends and validation data begins
    n = int(train_ratio * len(data))

    # Split the data into training and validation sets
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def create_data_batch(
    train_data, val_data, split, block_size=None, batch_size=None, device=None
):
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
    data = train_data if split == "train" else val_data

    indices = torch.randint(len(data) - block_size, (batch_size,))

    input_batch = torch.stack([data[i : i + block_size] for i in indices])

    target_batch = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])

    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    return input_batch, target_batch


if __name__ == "__main__":

    args = parse_arguments()

    path = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.data_dir.lstrip("/")
    )

    # print(path)

    data = load_data(path)

    # print(data[:100])
