from typing import Dict, List, Tuple
import re
import json
import string
from collections import Counter


def read_file(file_path: str, verbose: bool = False) -> List[str]:
    with open(file_path, "r") as file:
        text = file.read().lower()
        sentences = re.findall(r"\b\w+\b", text)
        if verbose:
            print(f"Read {len(sentences)} sentences from file.")
        return sentences


def get_vocab(text: List[str], verbose: bool = False) -> Dict[str, int]:
    vocab = Counter()
    for sentence in text:
        words = sentence.split()
        for word in words:
            # Treat each word as a sequence of individual characters
            vocab[" ".join(list(word))] += 1
    if verbose:
        print(f"Vocabulary size: {len(vocab)}")
    return vocab


def get_stats(
    vocab: Dict[str, int], verbose: bool = False
) -> Dict[Tuple[str, str], int]:
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    if verbose:
        print(f"Number of pairs: {len(pairs)}")
    return pairs


def merge_vocab(
    pair: Tuple[str, str], v_in: Dict[str, int], verbose: bool = False
) -> Dict[str, int]:
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    if verbose:
        print(f"Merged vocab size: {len(v_out)}")
    return v_out


def get_tokens_from_vocab(
    vocab: Dict[str, int], verbose: bool = False
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    tokens_frequencies = Counter()
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization["".join(word_tokens)] = word_tokens
    if verbose:
        print(f"Number of tokens: {len(tokens_frequencies)}")
    return tokens_frequencies, vocab_tokenization


def tokenize_bpe_word(word: str, vocab_tokenization: Dict[str, List[str]]) -> List[str]:
    if word in vocab_tokenization:
        return vocab_tokenization[word]

    tokens = []
    i = 0
    while i < len(word):
        max_token_length = len(word) - i
        subword = None
        for token_length in range(max_token_length, 0, -1):
            potential_subword = word[i : i + token_length]
            if potential_subword in vocab_tokenization:
                subword = potential_subword
                tokens.append(subword)
                break
        if (
            subword is None
        ):  # if no subword has been found, treat the character as a token
            tokens.append(word[i])
            i += 1
        else:
            i += len(subword)

    return tokens


def tokenize_bpe_sentence(
    sentence: str, vocab_tokenization: Dict[str, List[str]], verbose: bool = False
) -> List[str]:
    sentence = sentence.lower()  # ensure the sentence is in lower case
    sentence_tokens = []

    i = 0
    while i < len(sentence):
        max_subword = ""
        for j in range(i + 1, len(sentence) + 1):
            subword = sentence[i:j]
            if subword in vocab_tokenization and len(subword) > len(max_subword):
                max_subword = subword
        if max_subword:
            sentence_tokens.append(max_subword)
            i += len(max_subword)
        else:
            i += 1

    if verbose:
        print(f"Tokenized sentence: {sentence_tokens}")

    return sentence_tokens


def tokenize_bpe(
    text: List[str], num_merges: int, verbose: bool = False
) -> Tuple[Dict[str, int], List[List[str]]]:
    vocab = get_vocab(text)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
    tokenized_text = [tokenize_bpe_sentence(s, vocab_tokenization) for s in text]
    return tokens_frequencies, tokenized_text


def save_tokenizer(
    tokenizer: Tuple[Dict[str, int], List[List[str]]], save_path: str
) -> None:
    with open(save_path, "w") as file:
        json.dump(tokenizer, file)


def load_tokenizer(load_path: str) -> Tuple[Dict[str, int], List[List[str]]]:
    with open(load_path, "r") as file:
        tokenizer = json.load(file)
    return tokenizer


def main():
    file_path = "/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/datita.txt"  # replace with your file path
    num_merges = 50  # specify the number of merges during training
    save_path = "tokenizer.json"  # specify the path to save the tokenizer
    example_sentence = "This is an example sentence or and original sentence my king?"

    print("Original text: ", example_sentence)

    # Read the file
    sentences = read_file(file_path, verbose=True)

    # Train the tokenizer
    tokenizer = tokenize_bpe(sentences, num_merges, verbose=True)
    tokens_frequencies, vocab_tokenization = tokenizer

    print("\nAfter BPE Training:")
    print("Number of tokens: ", len(tokens_frequencies))
    print("Tokens and frequencies: ", tokens_frequencies)

    # Tokenize an example sentence
    tokenized_sentence = tokenize_bpe_sentence(
        example_sentence, vocab_tokenization, verbose=True
    )

    # Print tokenized sentence
    print("\nTokenized sentence:", tokenized_sentence)

    # Tokenize the file
    tokenized_file = [
        tokenize_bpe_sentence(sentence, vocab_tokenization) for sentence in sentences
    ]

    # Print tokenized file
    print("\nTokenized file:")
    for sentence in tokenized_file:
        print(sentence)

    # Save the tokenizer
    save_tokenizer(tokenizer, save_path)

    # Load the tokenizer
    loaded_tokenizer = load_tokenizer(save_path)
    loaded_tokens_frequencies, loaded_vocab_tokenization = loaded_tokenizer

    print("\nAfter Loading the Tokenizer:")
    print("Number of tokens: ", len(loaded_tokens_frequencies))
    print("Tokens and frequencies: ", loaded_tokens_frequencies)

    # Tokenize the example sentence using the loaded tokenizer
    re_tokenized_sentence = tokenize_bpe_sentence(
        example_sentence, loaded_vocab_tokenization, verbose=True
    )

    # Print re-tokenized sentence
    print("\nRe-tokenized sentence:", re_tokenized_sentence)


def test():
    example_sentence = "This is an example sentence, should we continue adding more chracters and see what it happens?"

    print("Original text: ", example_sentence)

    # Step 1: Initial vocabulary creation
    vocab = get_vocab([example_sentence])
    print("\nVocabulary size: ", len(vocab))
    print("Initial vocabulary: ", vocab)

    # Step 2: Calculating pair frequencies
    pairs = get_stats(vocab)
    print("\nNumber of pairs: ", len(pairs))
    print("Pair frequencies: ", pairs)

    # Step 3: Merging the most frequent pair
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print("\nMerged vocab size: ", len(vocab))
    print("Vocabulary after merge operation: ", vocab)

    # Step 4: Getting tokens and their frequencies
    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)
    print("\nNumber of tokens: ", len(tokens_frequencies))
    print("Tokens and frequencies: ", tokens_frequencies)
    print("\nVocabulary tokenization: ", vocab_tokenization)

    # Step 5: Tokenizing each word with current vocabulary
    words = example_sentence.split()
    for word in words:
        word_tokens = tokenize_bpe_word(word, vocab_tokenization)
        print("\nWord: ", word)
        print("Tokenized word: ", word_tokens)

    # Step 6: Tokenizing the whole sentence
    sentence_tokens = tokenize_bpe_sentence(example_sentence, vocab_tokenization)
    print("\nTokenized sentence: ", sentence_tokens)

    # Step 7: Full BPE tokenization (10 merges)
    tokens_frequencies, tokenized_text = tokenize_bpe([example_sentence], num_merges=10)
    print("\nFull BPE tokenization:")
    print("Token frequencies: ", tokens_frequencies)
    print("Tokenized text: ", tokenized_text)


if __name__ == "__main__":
    test()
