use std::env;

fn main() {
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=openblas");

    let sources = vec![
        "third_party/ACORN/faiss/AutoTune.cpp",
        "third_party/ACORN/faiss/Clustering.cpp",
        "third_party/ACORN/faiss/IVFlib.cpp",
        "third_party/ACORN/faiss/Index.cpp",
        "third_party/ACORN/faiss/Index2Layer.cpp",
        "third_party/ACORN/faiss/IndexAdditiveQuantizer.cpp",
        "third_party/ACORN/faiss/IndexBinary.cpp",
        "third_party/ACORN/faiss/IndexBinaryFlat.cpp",
        "third_party/ACORN/faiss/IndexBinaryFromFloat.cpp",
        "third_party/ACORN/faiss/IndexBinaryHNSW.cpp",
        "third_party/ACORN/faiss/IndexBinaryHash.cpp",
        "third_party/ACORN/faiss/IndexBinaryIVF.cpp",
        "third_party/ACORN/faiss/IndexFlat.cpp",
        "third_party/ACORN/faiss/IndexFlatCodes.cpp",
        "third_party/ACORN/faiss/IndexHNSW.cpp",
        "third_party/ACORN/faiss/IndexACORN.cpp",
        "third_party/ACORN/faiss/IndexIDMap.cpp",
        "third_party/ACORN/faiss/IndexIVF.cpp",
        "third_party/ACORN/faiss/IndexIVFAdditiveQuantizer.cpp",
        "third_party/ACORN/faiss/IndexIVFFlat.cpp",
        "third_party/ACORN/faiss/IndexIVFPQ.cpp",
        "third_party/ACORN/faiss/IndexIVFFastScan.cpp",
        "third_party/ACORN/faiss/IndexIVFAdditiveQuantizerFastScan.cpp",
        "third_party/ACORN/faiss/IndexIVFPQFastScan.cpp",
        "third_party/ACORN/faiss/IndexIVFPQR.cpp",
        "third_party/ACORN/faiss/IndexIVFSpectralHash.cpp",
        "third_party/ACORN/faiss/IndexLSH.cpp",
        "third_party/ACORN/faiss/IndexNNDescent.cpp",
        "third_party/ACORN/faiss/IndexLattice.cpp",
        "third_party/ACORN/faiss/IndexNSG.cpp",
        "third_party/ACORN/faiss/IndexPQ.cpp",
        "third_party/ACORN/faiss/IndexFastScan.cpp",
        "third_party/ACORN/faiss/IndexAdditiveQuantizerFastScan.cpp",
        "third_party/ACORN/faiss/IndexPQFastScan.cpp",
        "third_party/ACORN/faiss/IndexPreTransform.cpp",
        "third_party/ACORN/faiss/IndexRefine.cpp",
        "third_party/ACORN/faiss/IndexReplicas.cpp",
        "third_party/ACORN/faiss/IndexRowwiseMinMax.cpp",
        "third_party/ACORN/faiss/IndexScalarQuantizer.cpp",
        "third_party/ACORN/faiss/IndexShards.cpp",
        "third_party/ACORN/faiss/MatrixStats.cpp",
        "third_party/ACORN/faiss/MetaIndexes.cpp",
        "third_party/ACORN/faiss/VectorTransform.cpp",
        "third_party/ACORN/faiss/clone_index.cpp",
        "third_party/ACORN/faiss/index_factory.cpp",
        "third_party/ACORN/faiss/impl/AuxIndexStructures.cpp",
        "third_party/ACORN/faiss/impl/IDSelector.cpp",
        "third_party/ACORN/faiss/impl/FaissException.cpp",
        "third_party/ACORN/faiss/impl/HNSW.cpp",
        "third_party/ACORN/faiss/impl/ACORN.cpp",
        "third_party/ACORN/faiss/impl/NSG.cpp",
        "third_party/ACORN/faiss/impl/PolysemousTraining.cpp",
        "third_party/ACORN/faiss/impl/ProductQuantizer.cpp",
        "third_party/ACORN/faiss/impl/AdditiveQuantizer.cpp",
        "third_party/ACORN/faiss/impl/ResidualQuantizer.cpp",
        "third_party/ACORN/faiss/impl/LocalSearchQuantizer.cpp",
        "third_party/ACORN/faiss/impl/ProductAdditiveQuantizer.cpp",
        "third_party/ACORN/faiss/impl/ScalarQuantizer.cpp",
        "third_party/ACORN/faiss/impl/index_read.cpp",
        "third_party/ACORN/faiss/impl/index_write.cpp",
        "third_party/ACORN/faiss/impl/io.cpp",
        "third_party/ACORN/faiss/impl/kmeans1d.cpp",
        "third_party/ACORN/faiss/impl/lattice_Zn.cpp",
        "third_party/ACORN/faiss/impl/pq4_fast_scan.cpp",
        "third_party/ACORN/faiss/impl/pq4_fast_scan_search_1.cpp",
        "third_party/ACORN/faiss/impl/pq4_fast_scan_search_qbs.cpp",
        "third_party/ACORN/faiss/impl/io.cpp",
        "third_party/ACORN/faiss/impl/lattice_Zn.cpp",
        "third_party/ACORN/faiss/impl/NNDescent.cpp",
        "third_party/ACORN/faiss/invlists/BlockInvertedLists.cpp",
        "third_party/ACORN/faiss/invlists/DirectMap.cpp",
        "third_party/ACORN/faiss/invlists/InvertedLists.cpp",
        "third_party/ACORN/faiss/invlists/InvertedListsIOHook.cpp",
        "third_party/ACORN/faiss/utils/Heap.cpp",
        "third_party/ACORN/faiss/utils/WorkerThread.cpp",
        "third_party/ACORN/faiss/utils/distances.cpp",
        "third_party/ACORN/faiss/utils/distances_simd.cpp",
        "third_party/ACORN/faiss/utils/extra_distances.cpp",
        "third_party/ACORN/faiss/utils/hamming.cpp",
        "third_party/ACORN/faiss/utils/partitioning.cpp",
        "third_party/ACORN/faiss/utils/quantize_lut.cpp",
        "third_party/ACORN/faiss/utils/random.cpp",
        "third_party/ACORN/faiss/utils/sorting.cpp",
        "third_party/ACORN/faiss/utils/utils.cpp",
        "third_party/ACORN/faiss/utils/distances_fused/avx512.cpp",
        "third_party/ACORN/faiss/utils/distances_fused/distances_fused.cpp",
        "third_party/ACORN/faiss/utils/distances_fused/simdlib_based.cpp",
        "third_party/ACORN/faiss/IndexACORN.cpp",
    ];

    let mut build = cxx_build::bridge("src/lib.rs");

    for src in &sources {
        build.file(src);
    }

    build.include("third_party/faiss/");

    build.std("c++17");

    // Position-independent code
    build.flag("-fPIC");

    // NOTE: just for debugging
    build.flag("-w");

    if cfg!(target_os = "windows") {
        build.flag("/bigobj");
    } else {
        build.flag("-mavx2");
        build.flag("-mfma");
        build.flag("-mf16c");
        build.flag("-mpopcnt");
        build.flag("-fopenmp");
    }

    build.define("FINTEGER", Some("int"));

    if let Ok(mkl_lib) = env::var("MKL_LIBRARIES") {
        println!("cargo:rustc-link-lib={}", mkl_lib);
    } else if let Ok(lapack_lib) = env::var("LAPACK_LIBRARIES") {
        println!("cargo:rustc-link-lib={}", lapack_lib);
    }

    // Enable parallel builds
    // build.jobs(num_cpus::get());

    build.compile("oak");

    // println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");
    // for src in sources {
    //     println!("cargo:rerun-if-changed={}", src);
    // }
    // println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.cc");
    // println!("cargo:rerun-if-changed=third_party/ACORN/faiss/IndexACORN.h");
}
