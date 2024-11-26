# Use official rust image
FROM rust 

# Build ACORN
RUN apt-get update
RUN apt-get install cmake -y
RUN apt-get install git -y
RUN git clone -b oak https://github.com/csirianni/ACORN.git acorn

WORKDIR acorn

RUN apt install build-essential -y

RUN apt-get install libopenblas-dev -y
RUN apt-get install libgomp1 -y

# DEBUGGING
RUN apt-get install tmux -y
RUN apt-get install gdb -y


# Not totally clear why we still need these....
RUN cmake -B build -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
RUN cmake --build build
RUN cp /acorn/build/faiss/libfaiss.so /usr/local/lib/libfaiss.so
RUN cp /acorn/build/c_api/libfaiss_c.so /usr/local/lib/libfaiss_c.so
ENV LD_LIBRARY_PATH=/usr/local/lib

# Build oak
WORKDIR /oak
COPY . .
# RUN cargo build --release
# CMD ./oak
