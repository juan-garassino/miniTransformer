from your_package.tokenizer import Tokenizer


def test_tokenizer():
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Test sentence.")
    assert len(tokens) > 0  # Replace with more specific checks as needed
    # Check for specific tokens if possible
