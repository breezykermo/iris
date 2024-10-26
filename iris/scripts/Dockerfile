FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir faiss-cpu numpy

COPY compute_partitions.py /app
COPY data /app

CMD ["python", "compute_partitions.py", ""]

