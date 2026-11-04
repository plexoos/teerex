UV ?= uv
.DEFAULT_GOAL := help

.PHONY: help all install test build notebook clean distclean

help:
	@printf '%s\n' \
		'all        Build source and wheel distributions (alias for build)' \
		'install    Sync the project environment from uv.lock' \
		'test       Run the unittest suite' \
		'build      Build source and wheel distributions in dist/' \
		'notebook   Open the marimo analysis notebook' \
		'clean      Remove caches, package metadata, and build artifacts' \
		'distclean  Run clean and remove local .venv/ and venv/ environments'

all: build

install:
	$(UV) sync --locked

test:
	$(UV) run --locked python -m unittest discover -s tests -v

build:
	$(UV) build

notebook:
	$(UV) run --locked marimo edit notebooks/simload.py

# Preserve environments and study results during routine cleanup.
clean:
	find . -type d \( -name .git -o -name .venv -o -name venv \) -prune -o \
		-type d \( -name __pycache__ -o -name .ipynb_checkpoints \
		-o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \
		-o -name '*.egg-info' \) \
		-prune -exec rm -rf {} + -o \
		-type f \( -name '*.pyc' -o -name '*.pyo' -o -iname .DS_Store \) \
		-exec rm -f {} +
	rm -rf build dist

# Restore a checkout without generated files or local virtual environments.
distclean: clean
	rm -rf .venv venv
