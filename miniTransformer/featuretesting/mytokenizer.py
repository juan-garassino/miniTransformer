from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def load_text_file(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    return text


def train_tokenizer(text, num_merges):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Training
    trainer = trainers.BpeTrainer(vocab_size=num_merges)
    tokenizer.train_from_iterator([text], trainer)

    return tokenizer


def tokenize_text(text, tokenizer):
    encoded = tokenizer.encode(text)
    tokenized_sentences = [encoded.tokens]
    return tokenized_sentences


class MyTokenizer:

    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.tokenizer = None

    def train(self, text):
        self.tokenizer = train_tokenizer(text, self.num_merges)
        tokenized_sentences = tokenize_text(text, self.tokenizer)
        return tokenized_sentences

    def save_tokenizer(self, filepath):
        self.tokenizer.save(filepath)

    def load_tokenizer(self, filepath):
        self.tokenizer = Tokenizer.from_file(filepath)

    def tokenize_text_with_loaded_tokenizer(self, text):
        tokenized_sentences = tokenize_text(text, self.tokenizer)
        return tokenized_sentences

    def summary(self):
        vocab_size = len(self.tokenizer.get_vocab())
        print(f"Number of Tokens: {vocab_size}")
        print(self.tokenizer.get_vocab())

    def create_char_mappings(self):
        # Get the vocabulary of the tokenizer
        vocab = self.tokenizer.get_vocab()

        # Extract subwords from the vocabulary
        subwords = [
            subword for subword in vocab if not subword.startswith(" ")
        ]

        # Create mappings between subwords and integers
        subword_to_int = {subword: i for i, subword in enumerate(subwords)}
        int_to_subword = {i: subword for subword, i in subword_to_int.items()}

        return subword_to_int, int_to_subword, len(subwords)


if __name__ == '__main__':
    text = load_text_file(
        '/Users/juan-garassino/Code/juan-garassino/miniTransformer/miniTransformer/data/input.txt'
    )

    num_merges = 500

    tokenizer = MyTokenizer(num_merges)
    tokenized_text = tokenizer.train(text)

    # Save the tokenizer
    tokenizer.save_tokenizer('tokenizer.json')

    # Load the tokenizer
    tokenizer.load_tokenizer('tokenizer.json')

    # Tokenize text with the loaded tokenizer
    loaded_tokenized_text = tokenizer.tokenize_text_with_loaded_tokenizer(text)

    tokenizer.summary()

    text = '''
    Not all the water in the rough rude sea
    Can wash the balm off from an anointed king;
    The breath of worldly men cannot depose
    The deputy elected by the Lord:
    For every man that Bolingbroke hath pressed
    To lift shrewd steel against our golden crown,
    God for his Richard hath in heavenly pay
    A glorious angel: then, if angels fight,
    Weak men must fall, for heaven still guards the right.
    Welcome, my lord: how far off lies your power?
    '''

    tokenizer = MyTokenizer(num_merges=0)

    tokenizer.load_tokenizer('tokenizer.json')

    tokenized_text = tokenizer.tokenize_text_with_loaded_tokenizer(text)

    print(tokenized_text)
