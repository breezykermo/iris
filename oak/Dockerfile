# Use official rust image
FROM rust 

# Dependencies for ACORN
RUN apt-get update
# RUN apt-get install cmake -y
RUN apt-get install git -y
RUN apt-get install build-essential -y
RUN apt-get install libopenblas-dev -y
RUN apt-get install libgomp1 -y

# TOOLS FOR DEBUGGING
RUN apt-get install gdb -y

# ENV LD_LIBRARY_PATH=/usr/local/lib

# Build oak
WORKDIR /oak
COPY . .
# RUN cargo build --release
# CMD ./oak
