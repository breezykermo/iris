FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

# Download the SIFT dataset
# RUN wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
# RUN tar -xzvf sift.tar.gz
# RUN rm sift.tar.gz

RUN wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
RUN tar -xzvf siftsmall.tar.gz
RUN rm siftsmall.tar.gz


# Ensure outdir exists
RUN mkdir -p outdir

CMD ["python", "run.py"]

