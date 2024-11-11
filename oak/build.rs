fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("third_party/ACORN/faiss/IndexACORN.cc")
        .std("c++14")
        .compile("cxxbridge-demo");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.cc");
    println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.h");
}
