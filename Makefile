# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* miniTransformer/*.py

black:
	@black scripts/* miniTransformer/*.py miniTransformer/*/*.py

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

run_model:
	python -m miniTransformer.main

run_training:
	python -m miniTransformer.main \
	--batch_size 16 \
	--block_size 32 \
	--max_iters 1000 \
	--eval_interval 100 \
	--learning_rate 1e-3 \
	--device cpu \
	--eval_iters 10 \
	--n_embd 64 \
	--n_head 4 \
	--n_layer 4 \
	--dropout 0.0 \
	--colab 0 \
	--data_dir miniTransformer/miniTransformer/data \
	--name input.txt \
	--checkpoint_dir miniTransformer/miniTransformer/checkpoints \
	--save_interval 100 \
	--plots_dir miniTransformer/miniTransformer/heatmaps \
	--heatmap_interval 100

run_generation:
	python -m miniTransformer.main \
	--generate \
	--colab 0 \
	--checkpoint_dir miniTransformer/miniTransformer/checkpoints \
	--checkpoint checkpoint_999.pt \
	--n_of_char 2000

run_colab_training:
	python -m miniTransformer.main \
	--batch_size 16 \
	--block_size 32 \
	--max_iters 20000 \
	--eval_interval 100 \
	--learning_rate 1e-3 \
	--device cpu \
	--eval_iters 10 \
	--n_embd 64 \
	--n_head 4 \
	--n_layer 4 \
	--dropout 0.0 \
	--colab 1 \
	--data_dir miniTransformer/miniTransformer/data \
	--name input.txt \
	--checkpoint_dir miniTransformer/miniTransformer/checkpoints \
	--save_interval 100 \
	--plots_dir miniTransformer/miniTransformer/heatmaps \
	--heatmap_interval 100

run_colab_generation:
	python -m miniTransformer.main \
	--generate \
	--colab 0 \
	--checkpoint_dir miniTransformer/miniTransformer/checkpoints \
	--checkpoint checkpoint_999.pt \
	--n_of_char 2000

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
