FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libwebp-dev \
    curl \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Install CPU version by default
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU support via build argument
ARG BUILD_GPU=false
RUN if [ "$BUILD_GPU" = "true" ]; then \
    pip uninstall -y torch torchvision && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
    fi

COPY app ./app
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
