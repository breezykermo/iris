// extern crate libc;
use crate::architecture::HardwareArchitecture;
use crate::stubs::gen_random_vecs;
use std::ffi::c_void;
use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use memmap2::Mmap;

/// Trait for a dataset of vectors.
pub trait Dataset: Sized {
    /// Create a new dataset, loading into memory or keeping on disk as per `hw_arch`.
    fn new(hw_arch: HardwareArchitecture) -> Result<Self>;
    /// Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    /// Returns None if the data is not yet trained, else the HardwareArchitecture on which it was
    /// trained.
    fn get_hardware_architecture(&self) -> HardwareArchitecture;
    // Returns data in dataset. Fails if full dataset doesn't fit in memory.
    // Only needs to be implemented as a way to test.
    // fn get_data(&self) -> Result<Vectors>;
}

enum VectorIndex {
    L2Flat,
}

pub trait Searchable {
    fn new(ds: impl Dataset, idx: VectorIndex);
    fn search(&self, query_vectors: Vectors, topk: Option<usize>);
}

///       32bit          ⟨d⟩ * 32bits
/// |--------------|-------/ ... /-------|
/// ⟨d⟩ dimension
pub struct Fvec {
    dimensionality: u32,
    data: Box<[f32]>,
}

/// Contiguous array of Fvecs
pub struct Vectors {
    raw_data: Box<[Fvec]>,
}

/// Deep1X Dataset implementation
pub struct Deep1X {
    mmap: Mmap,
    hw_arch: HardwareArchitecture,
}

impl Dataset for Deep1X {
    fn new(hw_arch: HardwareArchitecture) -> Result<Self> {
        // TODO: ensure filename formula maps to the output of Python provisioning
        // using `architecture`.
        let mut fname = PathBuf::new();
        fname.push("../data/deep1k.fvecs");

        let f = File::open(fname)?;

        let mmap = unsafe { Mmap::map(&f)? };

        Ok(Self { mmap, hw_arch })
    }

    fn dataset_info(&self) -> String {
        "Section of the Deep1B dataset X vectors".to_string()
    }

    fn get_hardware_architecture(&self) -> HardwareArchitecture {
        self.hw_arch
    }

    // fn get_data(&self) -> Result<Vectors> {
    //     unimplemented!()
    //     // let d = 64;
    //     let raw_slice = &self.mmap[..];
    //     // let index = faiss::IndexL2(d);
    //     // index.add(1000, raw_slice)
    //     // Ok(vecs)
    // }
}

impl Searchable for Deep1X {
    fn new(ds: impl Dataset, idx: VectorIndex) {
        // properly instantiate the faiss index (via FFI) according to `VectorIndex`
        // index = faiss.IndexFlatL2(d)

        // add all vectors in `ds` to the index
        // index.add(xb)
        // When calling `add` in the c_api, we need to pass two arguments:
        // 1) the number of vectors to add.
        // 2) a pointer to the array of vectors to be added

        // Assumptions on the data structure of argument 2:
        // NOTE: in IndexFlatCodes.cpp, the `add` method calls `sa_encode` with a pointer to the end
        // of the array where the vectors should be added.
        // In IndexFlat.cpp, there is an impl of `sa_encode` that looks like:
        // memcpy(bytes, x, sizeof(float) * d * n);
        // From this, we are assuming that when implementing the FFI binding for faiss' `add`
        // method, it is sufficient to pass a slice of the fvec bytes, i.e. each vector
        // encoded with its dimensionality first as a u32, then its vectors (see `Fvec` docs), as
        // `add`s second argument.
    }

    fn search(&self, query_vectors: Vectors, topk: Option<usize>) {
        // _, ids = index.search(x=xq, k=topk)
    }
}
