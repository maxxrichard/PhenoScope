.PHONY: install test lint format serve demo docker

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests pipelines

format:
	ruff check --fix src tests pipelines

serve:
	phenoscope serve --reload

demo:
	phenoscope demo

docker:
	docker build -f deploy/docker/Dockerfile -t phenoscope:latest .
	docker build -f deploy/docker/Dockerfile.gpu -t phenoscope-gpu:latest .
