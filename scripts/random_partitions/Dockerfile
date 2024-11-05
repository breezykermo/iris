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

# Docker command should be specified when running, i.e.
# 
# docker run -v ./outdir:/app/outdir compute_partitions python compute_partitions.py deep1k --out outdir 

