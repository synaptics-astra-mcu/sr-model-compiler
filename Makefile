.PHONY: lint check_format format test check all

all: check_format lint test

lint:
	pylint src/sr_model_compiler/sr_model_compiler.py src/sr_model_compiler/sr100_model_optimizer.py tests
	#pylint src/sr_model_compiler

check_format:
	black --check src/sr_model_compiler tests

format:
	black --line-length 88 src/sr_model_compiler tests

test:
	pytest tests
