SHELL = /bin/bash
.PHONY: help _prep create_environment requirements format lint style clean \
    clean-build clean-pyc clean-test docs docs-serve test setup-precommit

## GLOBALS
PROJECT_NAME = mlops-versioning
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python3

###############################################################################
# HELP
###############################################################################

help:
	@echo "Commands:"
	@echo "  setup-precommit    : Install and configure pre-commit hooks"
	@echo "  create_environment : Create conda environment"
	@echo "  requirements       : Install Python dependencies"
	@echo "  style              : Format code with black, isort"
	@echo "  lint               : Check code style and quality"
	@echo "  clean              : Remove temporary files and caches"
	@echo "  clean-all          : Remove all build, test, and cache files"
	@echo "  docs               : Build documentation"
	@echo "  docs-serve         : Serve documentation locally"
	@echo "  test               : Run tests"

###############################################################################
# SETUP
###############################################################################

## Set up python interpreter environment
create_environment:
	conda create -f environment.yaml -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements.txt

## Setup pre-commit hooks
setup-precommit:
	$(PYTHON_INTERPRETER) -m pip install pre-commit
	pre-commit install
	pre-commit autoupdate
	@echo ">>> Pre-commit hooks installed successfully"
	@echo ">>> Run 'pre-commit run --all-files' to test"

###############################################################################
# DEVELOPMENT
###############################################################################

## Format the code using isort and black
format:
	$(PYTHON_INTERPRETER) -m isort --profile black src/ scripts/ notebooks/
	$(PYTHON_INTERPRETER) -m black --line-length 100 src/ scripts/ notebooks/

## Lint code with flake8, isort, and black
lint:
	$(PYTHON_INTERPRETER) -m flake8 src/ scripts/ --max-line-length=100 --ignore=E203,W503
	$(PYTHON_INTERPRETER) -m isort --check --profile black src/ scripts/ notebooks/
	$(PYTHON_INTERPRETER) -m black --check --line-length 100 src/ scripts/ notebooks/

## Style: Format and check code
style: format lint

###############################################################################
# CLEANING
###############################################################################

## Remove temporary files and caches
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.DS_Store" -ls -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage* htmlcov/ .tox/
	@echo "✓ Cleanup complete"

## Remove build artifacts
clean-build:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.egg' -exec rm -f {} + 2>/dev/null || true
	@echo "✓ Build cleanup complete"

## Remove Python file artifacts
clean-pyc:
	@echo "Cleaning Python artifacts..."
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Python artifacts cleaned"

## Remove test and coverage artifacts
clean-test:
	@echo "Cleaning test artifacts..."
	rm -rf .tox/
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	@echo "✓ Test artifacts cleaned"

## Complete cleanup
clean-all: clean clean-build clean-pyc clean-test
	@echo "✓ Complete cleanup finished"

###############################################################################
# DOCUMENTATION
###############################################################################

## Build documentation
docs:
	@echo "Building documentation..."
	cd docs && mkdocs build
	@echo "✓ Documentation built"

## Serve documentation locally
docs-serve:
	@echo "Serving documentation at http://127.0.0.1:8000"
	cd docs && mkdocs serve

###############################################################################
# TESTING
###############################################################################

## Run tests
test:
	@echo "Running tests..."
	pytest -vv tests/
	@echo "✓ Tests complete"

## Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term tests/
	@echo "✓ Coverage report generated in htmlcov/"

###############################################################################
# PRE-COMMIT
###############################################################################

## Run pre-commit on all files
precommit-all:
	pre-commit run --all-files

## Update pre-commit hooks
precommit-update:
	pre-commit autoupdate
