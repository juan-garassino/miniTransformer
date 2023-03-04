import tensorflow as tf
import re


class SimpleTokenizer:

    def __init__(self):
        self.vocab = {}
        self.tokenizer_re = re.compile(r"\w+|\S")

    def fit(self, texts):
        for text in texts:
            for token in self.tokenizer_re.findall(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        return [self.vocab[token] for token in self.tokenizer_re.findall(text)]

    def decode(self, ids):
        return "".join(
            list(self.vocab.keys())[list(self.vocab.values()).index(id)]
            for id in ids)


def create_dataset(file_path, batch_size, seq_length):
    # Load the text file and convert to integer ids
    tokenizer = SimpleTokenizer()
    with open(file_path, 'r') as f:
        texts = f.readlines()
        tokenizer.fit(texts)
        data = [tokenizer.encode(text) for text in texts]
    vocab = tokenizer.vocab
    char2idx = {char: idx for idx, char in vocab.items()}
    idx2char = {idx: char for idx, char in vocab.items()}

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(tf.concat(data, axis=0))

    # Split the data into sequences of a fixed length
    sequences = dataset.batch(seq_length + 1, drop_remainder=True)

    # Define the input and output sequences
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Create batches of data
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the data for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, vocab, char2idx, idx2char, tokenizer
