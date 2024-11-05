# Use official rust image
FROM rust 

# Build faiss
RUN apt-get update
RUN apt-get install cmake -y
RUN apt-get install git -y
RUN git clone https://github.com/Enet4/faiss.git

WORKDIR faiss

RUN apt install build-essential -y

RUN apt-get install libopenblas-dev -y

# Disable GPU because we don't want to install CUDA
RUN cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
RUN cmake --build build
RUN cp /faiss/build/faiss/libfaiss.so /usr/local/lib/libfaiss.so
RUN cp /faiss/build/c_api/libfaiss_c.so /usr/local/lib/libfaiss_c.so
ENV LD_LIBRARY_PATH=/usr/local/lib

# Build oak
WORKDIR /oak
COPY . .
# RUN cargo build --release
# CMD ./oak
