.PHONY: help install install-coco train train-coco run test download-coco preprocess-coco setup-coco quick-start quick-start-coco build build-cpu build-gpu build-all build-hub-cpu build-hub-gpu build-hub-all push-hub docker-run docker-run-gpu docker-stop docker-test clean format lint

REGISTRY ?= docker.io
DOCKER_IMAGE ?= atakanemree/image-caption-api

# Default target
help:
	@echo "Image Captioning API - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  install     Install Python dependencies"
	@echo "  train       Train the sample model"
	@echo "  train-coco  Train on COCO (after setup)"
	@echo "  run         Run the API server locally"
	@echo "  test        Run smoke tests against local server"
	@echo "  install-coco Install COCO training dependencies"
	@echo ""
	@echo "COCO Dataset:"
	@echo "  download-coco    Download COCO 2017 dataset"
	@echo "  preprocess-coco  Process COCO data and create splits"
	@echo "  setup-coco       Download and preprocess COCO"
	@echo "  quick-start-coco Install COCO deps + setup + train"
	@echo ""
	@echo "Docker:"
	@echo "  build           Build standard Docker image"
	@echo "  build-cpu       Build CPU-only Docker image"
	@echo "  build-gpu       Build GPU-enabled Docker image"
	@echo "  build-hub-cpu   Build Docker Hub optimized CPU image"
	@echo "  build-hub-gpu   Build Docker Hub optimized GPU image"
	@echo "  build-all       Build all Docker variants"
	@echo "  push-hub        Push hub images (override DOCKER_IMAGE)"
	@echo "  docker-run      Run standard Docker container"
	@echo "  docker-run-gpu  Run GPU Docker container"
	@echo "  docker-stop     Stop running Docker container"
	@echo "  docker-test     Test Docker container"
	@echo ""
	@echo "Code Quality:"
	@echo "  format      Format Python code"
	@echo "  lint        Run linting checks"

# Development commands
install:
	pip install -r requirements.txt
install-coco:
	pip install -r requirements-coco.txt

train:
	cd training && python train.py

train-coco:
	cd training && python train_coco.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "Starting API server in background..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
	echo $$! > .api_server.pid; \
	sleep 5; \
	python test_api.py; \
	kill $$(cat .api_server.pid) 2>/dev/null || true; \
	rm -f .api_server.pid

# COCO dataset commands
download-coco:
	@echo "Downloading COCO dataset..."
	python scripts/download_coco.py --data-dir ./data

preprocess-coco:
	@echo "Preprocessing COCO dataset..."
	python scripts/preprocess_coco.py --data-dir ./data --output-dir ./data/processed

setup-coco: download-coco preprocess-coco
	@echo "COCO dataset setup completed!"

quick-start-coco: install-coco setup-coco train-coco
	@echo "COCO training completed!"

# Docker commands
build:
	docker build -t image-caption-api .

build-hub-cpu:
	docker build -f Dockerfile.hub --build-arg BUILD_GPU=false -t $(DOCKER_IMAGE):cpu .

build-hub-gpu:
	docker build -f Dockerfile.hub --build-arg BUILD_GPU=true -t $(DOCKER_IMAGE):gpu .

build-hub-all: build-hub-cpu build-hub-gpu

build-gpu:
	docker build --build-arg BUILD_GPU=true -t image-caption-api:gpu .

build-cpu:
	docker build -f Dockerfile.cpu -t image-caption-api:cpu .

build-all: build build-cpu build-gpu build-hub-cpu build-hub-gpu
	@echo "Built all Docker variants: standard, gpu, cpu"

push-hub: build-hub-all
	docker push $(DOCKER_IMAGE):cpu
	docker push $(DOCKER_IMAGE):gpu

docker-run:
	@echo "Starting API in Docker container..."
	docker run -d --name caption-api -p 8000:8000 image-caption-api
	@echo "Container started. API available at http://localhost:8000"
	@echo "Stop with: docker stop caption-api"

docker-run-gpu:
	@echo "Starting GPU API in Docker container..."
	docker run -d --gpus all --name caption-api-gpu -p 8001:8000 image-caption-api:gpu
	@echo "GPU container started. API available at http://localhost:8001"
	@echo "Stop with: docker stop caption-api-gpu"

docker-stop:
	docker stop caption-api caption-api-gpu caption-api-test 2>/dev/null || true
	docker rm caption-api caption-api-gpu caption-api-test 2>/dev/null || true

docker-test: build-cpu
	@echo "Testing Docker container..."
	@docker run -d --name caption-api-test -p 8001:8000 image-caption-api:cpu; \
	sleep 10; \
	API_BASE_URL=http://localhost:8001 python test_api.py; \
	docker stop caption-api-test; \
	docker rm caption-api-test

# Maintenance commands
clean:
	@echo "Cleaning generated files..."
	@rm -f .api_server.pid
	@rm -rf training/sample_images
	@rm -f training/sample_captions.json
	@rm -rf __pycache__ app/__pycache__ training/__pycache__
	@rm -f *.pyc **/*.pyc
	@echo "Clean complete."

format:
	@echo "Formatting Python code..."
	@black app/ training/ test_api.py 2>/dev/null || echo "Install black with: pip install black"

lint:
	@echo "Running linting checks..."
	@flake8 app/ training/ test_api.py 2>/dev/null || echo "Install flake8 with: pip install flake8"

# Quick start
quick-start: install train test
	@echo "🎉 Quick start complete! API is ready for development."
