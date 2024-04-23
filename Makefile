# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* miniTransformer/*.py

black:
	@black scripts/* *.py */*.py */*/*.py */*/*/*.py 
	
syntax:
	@pylint *.py */*.py */*/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr miniTransformer-*.dist-info
	@rm -fr miniTransformer.egg-info
	@rm -fr results/*

run_sourcing:
	poetry run python -m miniTransformer.sourcing.sourcing \
	--root_dir Code/juan-garassino/miniNetworks/ \
	--data_dir miniTransformer/data

run_simple_tokenizer:
	poetry run python -m miniTransformer.preprocessing.tokenizers.simple_tokenizer \
	--root_dir Code/juan-garassino/miniNetworks/ \
	--data_dir miniTransformer/data

run_regex_tokenizer:
	poetry run python -m miniTransformer.preprocessing.tokenizers.regex_tokenizer \
	--root_dir Code/juan-garassino/miniNetworks/ \
	--data_dir miniTransformer/data

run_training:
	poetry run python -m miniTransformer.main \
	--root_dir $(ROOT_DIR) \
	--batch_size 16 \
	--block_size 32 \
	--vocab_size 260 \
	--max_iters 500 \
	--eval_interval 100 \
	--learning_rate 1e-3 \
	--device cpu \
	--eval_iters 100 \
	--embd_dim 64 \
	--n_head 4 \
	--n_layer 4 \
	--dropout 0.0 \
	--data_dir $(DATA_DIR) \
	--checkpoints_dir $(CHECKPOINTS_DIR) \
	--tokenizers_dir $(TOKENIZERS_DIR) \
	--save_interval 100 \
	--heatmaps_dir $(HEATMAPS_DIR) \
	--heatmap_interval 25 \
	--animations_dir $(ANIMATIONS_DIR) \
	--tokenizer regex

run_generation:
	poetry run python -m miniTransformer.main \
	--root_dir code/juan-garassino/miniNetworks/ \
	--generate \
	--colab 0 \
	--checkpoints_dir miniTransformer/results/checkpoints \
	--checkpoint checkpoint_99.pt \
	--n_of_char 2000 \
	--tokenizer regex

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
