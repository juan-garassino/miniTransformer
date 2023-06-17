import re
import collections
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.lower().strip() for line in file]


def tokenize_sentence(sentence: str) -> List[str]:
    return re.findall(r"\b\w+\b|\S", sentence)


def get_vocab(text: List[str]) -> Dict[str, int]:
    return collections.Counter(
        " ".join(list(word)) + " </w>"
        for sentence in text
        for word in tokenize_sentence(sentence)
    )


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pairs[tokens[i], tokens[i + 1]] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    return {
        re.sub(
            "(?<!\S)" + re.escape(" ".join(pair)) + "(?!\S)", "".join(pair), word
        ): freq
        for word, freq in v_in.items()
    }


def get_tokens_and_text(
    file_path: str, num_merges: int = 10000
) -> Tuple[Dict[str, int], List[List[int]]]:
    text = read_file(file_path)
    vocab = get_vocab(text)
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    tokens_frequencies, vocab_tokenization = collections.defaultdict(int), {}
    for word, freq in vocab.items():
        tokens = word.split()
        for token in tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization["".join(tokens).replace("</w>", "")] = tokens
    token_to_id = {token: i for i, token in enumerate(tokens_frequencies.keys())}
    token_to_id["<unk>"] = len(token_to_id)
    tokenized_text = [
        [
            token_to_id.get(token, token_to_id["<unk>"])
            for word in tokenize_sentence(sentence)
            for token in vocab_tokenization.get(word, [word])
        ]
        for sentence in text
    ]
    return token_to_id, tokenized_text


def co_occurrence_matrix(
    tokenized_text: List[List[int]], vocab: Dict[str, int], window_size: int
) -> np.ndarray:
    vocab_size = len(vocab)
    co_occurrence_mat = np.zeros((vocab_size, vocab_size))
    for sentence in tokenized_text:
        sentence_length = len(sentence)
        for i, word in enumerate(sentence):
            start_idx = max(i - window_size, 0)
            end_idx = min(i + window_size + 1, sentence_length)
            context_words = sentence[start_idx:i] + sentence[i + 1 : end_idx]
            for context_word in context_words:
                co_occurrence_mat[word, context_word] += 1
    return co_occurrence_mat


def train_word_embeddings(
    tokenized_text: List[List[int]],
    vocab: Dict[str, int],
    embedding_dim: int,
    window_size: int,
    learning_rate: float,
    num_epochs: int,
) -> Dict[str, np.ndarray]:
    vocab_size = len(vocab)
    co_occurrence_mat = co_occurrence_matrix(tokenized_text, vocab, window_size)

    # Convert the co-occurrence matrix to a PyTorch tensor
    co_occurrence_mat = torch.FloatTensor(co_occurrence_mat)

    # Initialize word embeddings using PyTorch
    word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    word_embeddings.weight.data.uniform_(-1, 1)

    # Define the optimizer and loss function
    optimizer = optim.Adam(word_embeddings.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input_ids = []
        output_ids = []
        co_occurrences = []
        for i in range(vocab_size):
            for j in range(vocab_size):
                if co_occurrence_mat[i, j] > 0:
                    input_ids.append(i)
                    output_ids.append(j)
                    co_occurrences.append(co_occurrence_mat[i, j])
        input_ids = torch.LongTensor(input_ids)
        output_ids = torch.LongTensor(output_ids)
        co_occurrences = torch.FloatTensor(co_occurrences)

        input_embeds = word_embeddings(input_ids)
        output_embeds = word_embeddings(output_ids)
        pred_co_occurrences = torch.sum(input_embeds * output_embeds, dim=1)
        loss = criterion(pred_co_occurrences, co_occurrences)
        loss.backward()
        optimizer.step()

    # Retrieve the learned embeddings
    learned_embeddings = word_embeddings.weight.detach().numpy()

    # Create a dictionary mapping tokens to their embeddings
    token_to_embedding = {
        token: learned_embeddings[token_id] for token, token_id in vocab.items()
    }

    return token_to_embedding


if __name__ == "__main__":
    file_path = "data.txt"
    num_merges = 10000
    window_size = 2
    embedding_dim = 100
    learning_rate = 0.001
    num_epochs = 10

    # Tokenize the text
    token_to_id, tokenized_text = get_tokens_and_text(file_path, num_merges)

    # Create the co-occurrence matrix
    co_occurrence_mat = co_occurrence_matrix(tokenized_text, token_to_id, window_size)

    # Train word embeddings
    word_embeddings = train_word_embeddings(
        tokenized_text,
        token_to_id,
        embedding_dim,
        window_size,
        learning_rate,
        num_epochs,
    )

    # Access the embeddings of specific tokens
    print(word_embeddings.keys())

    # Save the word embeddings to a file
    np.save("word_embeddings.npy", word_embeddings)
