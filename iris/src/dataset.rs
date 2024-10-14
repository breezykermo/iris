// extern crate libc;
use crate::architecture::HardwareArchitecture::{
    self, DramBalancedPartitioning, DramRandomPartitioning,
};
use crate::stubs::gen_random_vecs;
// use std::ffi::c_void;
use std::fs::File;
use std::path::PathBuf;
use std::slice;

use anyhow::Result;
use memmap2::Mmap;
use std::marker::PhantomData;

const FOUR_BYTES: usize = std::mem::size_of::<f32>();

/// Trait for a dataset of vectors.
pub trait Dataset: Sized {
    /// Create a new dataset, loading into memory or keeping on disk as per `hw_arch`.
    fn new(hw_arch: HardwareArchitecture) -> Result<Self>;
    /// Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    /// Returns None if the data is not yet trained, else the HardwareArchitecture on which it was
    /// trained.
    fn get_hardware_architecture(&self) -> HardwareArchitecture;
    /// Returns data in dataset. Fails if full dataset doesn't fit in memory.
    fn get_data(&self) -> Result<Vec<Fvec>>;
}

enum VectorIndex {
    L2Flat,
}

pub trait Searchable {
    fn new(ds: impl Dataset, idx: VectorIndex);
    fn search(&self, query_vectors: Vec<Fvec>, topk: Option<usize>);
}

/// A type that represents a view on an underlying set of fvecs. See [FVecView] for memory layout.
pub struct FvecsView<'a> {
    ptr: *const u8,
    end: *const u8,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> FvecsView<'a> {
    pub fn new(mmap: &Mmap) -> Self {
        let ptr = mmap.as_ptr();
        FvecsView {
            ptr,
            // SAFETY: the pointer arithmetic is constrained by the length of the file (represented
            // by `mmap`)
            end: unsafe { ptr.add(mmap.len()) },
            _marker: PhantomData,
        }
    }
}

impl<'a> Iterator for FvecsView<'a> {
    type Item = FvecView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            return None;
        }

        // Ensure that there is at least a `dimensionality` value left in the file.
        if (self.end as usize) - (self.ptr as usize) < FOUR_BYTES {
            return None;
        }

        // Read the next 4 bytes as an u32.
        // SAFETY: We know that there are at least four bytes left in the file from the check
        // directly above.
        let dimensionality = unsafe {
            let value = *(self.ptr as *const u32);
            self.ptr = self.ptr.add(FOUR_BYTES);
            value
        };

        if dimensionality <= 0 {
            // A vector with dimensionality of 0 is impossible, I think?
            return None;
        }

        let dimensionality = dimensionality as usize;
        let span = dimensionality * FOUR_BYTES;

        let ptr = self.ptr as *const f32;

        // SAFETY: We assume from the dimensionality vector that there are `span` bytes to read.
        self.ptr = unsafe { self.ptr.add(span) };

        Some(FvecView {
            dimensionality,
            ptr,
            _marker: PhantomData,
        })
    }
}

/// A type that represents a view on an underlying fvec that 'lazily' determines values (i.e. does
/// not copy bytes up front).
///
/// The layout of an fvec on disk is as follows:
///
///       32bit          ⟨d⟩ * 32bits
/// |--------------|-------/ ... /-------|
///
/// where ⟨d⟩ is the dimensionality of the vector.
///
/// The `dimensionality` has already been parsed: so the `ptr` begins where the f32 data begins.
pub struct FvecView<'a> {
    dimensionality: usize,
    ptr: *const f32,
    _marker: PhantomData<&'a [u8]>,
}

pub struct Fvec {
    dimensionality: usize,
    data: Vec<f32>,
}

impl<'a> From<FvecView<'a>> for Fvec {
    fn from(value: FvecView) -> Self {
        // SAFETY: We assume that the view has been well constructed.
        let data: &[f32] = unsafe { slice::from_raw_parts(value.ptr, value.dimensionality) };
        Self {
            dimensionality: value.dimensionality,
            data: data.to_vec(),
        }
    }
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

        // SAFETY: For the purposes of our benchmarking suite, we are assuming that the underlying
        // file will not be modified throughout the duration of the program, as we control the file
        // system.
        let mmap = unsafe { Mmap::map(&f)? };

        if let DramBalancedPartitioning | DramRandomPartitioning = hw_arch {
            // Calls syscall mlock on file memory, ensuring that it will be in RAM until unlocked.
            // This will through an error if RAM is not large enough.
            let _ = mmap.lock()?;
        }

        Ok(Self { mmap, hw_arch })
    }

    fn dataset_info(&self) -> String {
        "Section of the Deep1B dataset X vectors".to_string()
    }

    fn get_hardware_architecture(&self) -> HardwareArchitecture {
        self.hw_arch
    }

    fn get_data(&self) -> Result<Vec<Fvec>> {
        let view = FvecsView::new(&self.mmap);
        let vecs = view.map(|v| Fvec::from(v)).collect();
        Ok(vecs)
    }
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

    fn search(&self, query_vectors: Vec<Fvec>, topk: Option<usize>) {
        // _, ids = index.search(x=xq, k=topk)
    }
}
