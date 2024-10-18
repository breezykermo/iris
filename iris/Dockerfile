# Use Ubuntu 22.04 for rust build layer
# FROM ubuntu AS rust_build_layer
FROM rust AS rust_build_layer

WORKDIR /build

# Compile openssl for static linking
# RUN dnf -y install gcc-c++ pkg-config musl-gcc git perl-core binaryen
# RUN git clone git://git.openssl.org/openssl.git
# RUN cd openssl && git checkout OpenSSL_1_1_1-stable
# RUN cd openssl && ./config -fPIC no-weak-ssl-ciphers no-async --prefix=/usr/local/ssl --openssldir=/usr/local/ssl
# RUN cd openssl && make && make install
# ENV OPENSSL_STATIC true
# ENV OPENSSL_DIR /usr/local/ssl

# Install rust
# RUN curl --proto '=https' --tlsv1.3 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"
# TODO: Do we need this?
# RUN rustup toolchain install 1.72.0
# RUN rustup toolchain install nightly-2023-08-01

# Add compilation targets
# RUN rustup target add x86_64-unknown-linux-gnu

COPY . .
RUN cargo build --release

# Runtime Docker container 
FROM ubuntu 
COPY --from=rust_build_layer /build/target/release /iris
WORKDIR /iris
CMD ./iris
