fn main() {
    println!("cargo:rustc-link-lib=dylib=faiss");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-search=native=/usr/local/lib");

    cxx_build::bridge("src/lib.rs")
        // .include("third_party/ACORN/faiss/impl")
        .file("third_party/ACORN/faiss/IndexACORN.cpp")
        .define("FINTEGER", "int")
        .std("c++14")
        .compile("cxxbridge-demo");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.cc");
    println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.h");
}
