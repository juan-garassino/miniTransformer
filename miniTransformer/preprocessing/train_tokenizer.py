"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from miniTransformer.preprocessing.tokenizers.basic_tokenizer import BasicTokenizer
from miniTransformer.preprocessing.tokenizers.regex_tokenizer import RegexTokenizer
from miniTransformer.utils.parse_arguments import parse_arguments

# Get the project root directory from the environment variable
args = parse_arguments()

project_root = os.path.join(
        os.environ.get("HOME"), args.root_dir, args.data_dir.lstrip("/")
    )
# Construct the path to the data file
data_file_path = os.path.join(project_root, "data.txt")

# Open the file and read its contents
with open(data_file_path, "r", encoding="utf-8") as file:
    text = file.read()

results_file_path = os.path.join(project_root, "results", "tokenizers_models")

# create a directory for models, so we don't pollute the current directory
os.makedirs(results_file_path, exist_ok=True)

t0 = time.time()

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join(results_file_path, name)
    tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
