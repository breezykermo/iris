// extern crate libc;
// use std::ffi::c_void;
use crate::ffi;
use std::convert::TryFrom;
use std::fs::File;
use std::path::PathBuf;
use std::slice;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "hnsw_faiss")]
use faiss::index::IndexImpl;
#[cfg(feature = "hnsw_faiss")]
use faiss::{index_factory, Index, MetricType};
use memmap2::Mmap;
use std::marker::PhantomData;
use thiserror::Error;

const FOUR_BYTES: usize = std::mem::size_of::<f32>();

/// Trait for a dataset of vectors.
/// Note that this must be `Sized` in order that the constructor can return a Result.
pub trait Dataset: Sized {
    /// Create a new dataset, loading into memory.
    fn new(fname: String) -> Result<Self>;
    /// Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    /// Provide the dimensionality of the vectors in the dataset.
    fn get_dimensionality(&self) -> u32;
    /// Returns data in dataset. Fails if full dataset doesn't fit in memory.
    fn get_data(&self) -> Result<Vec<Fvec>>;
}

// https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
#[derive(Debug)]
pub enum VectorIndex {
    IndexFlatL2,
    HNSWAcornFlat,
}

#[derive(Debug)]
pub enum VectorMetric {
    IndexFlatL2,
}

impl ToString for VectorIndex {
    fn to_string(&self) -> String {
        // String is passed to FAISS, and thus should match:
        // https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        match self {
            VectorIndex::IndexFlatL2 => "Flat".to_string(),
            VectorIndex::HNSWAcornFlat => "HNSW,Acorn".to_string(),
        }
    }
}

pub trait Searchable {
    /// Takes a Vec<Fvec> and returns a Vec<Vec<usize>>, whereby each inner Vec<usize> is an array
    /// of the indices for the `topk` vectors returned from the result.
    fn search(
        &self,
        query_vectors: Vec<Fvec>,
        topk: Option<usize>,
    ) -> Result<Vec<Vec<usize>>, SearchableError>;
}

/// A type that represents a view on an underlying set of fvecs. See [FVecView] for memory layout.
pub struct FvecsView<'a> {
    ptr: *const f32,
    end: *const f32,
    _marker: PhantomData<&'a [f32]>,
}

impl<'a> FvecsView<'a> {
    pub fn new(mmap: &Mmap) -> Self {
        let ptr = mmap.as_ptr() as *const f32;
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
    _marker: PhantomData<&'a [f32]>,
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

pub struct FlattenedVecs {
    dimensionality: usize,
    // This data is the flattened representation of all vectors of size `dimensionality`.
    data: Vec<f32>,
}

impl<'a> From<FvecsView<'a>> for FlattenedVecs {
    fn from(value: FvecsView) -> Self {
        // Assumptions on the data structure of argument 2:
        // NOTE: in IndexFlatCodes.cpp, the `add` method calls `sa_encode` with a pointer to the end
        // of the array where the vectors should be added.
        // In IndexFlat.cpp, there is an impl of `sa_encode` that looks like:
        // memcpy(bytes, x, sizeof(float) * d * n);
        // From this, we are assuming that when implementing the FFI binding for faiss' `add`
        // method, it is sufficient to pass a slice of the fvec bytes, i.e. each vector
        // encoded with its dimensionality first as a u32, then its vectors (see `Fvec` docs), as
        // `add`s second argument.

        value.fold(
            FlattenedVecs {
                dimensionality: 0,
                data: Vec::new(),
            },
            |mut acc, v| {
                if acc.dimensionality == 0 {
                    acc.dimensionality = v.dimensionality;
                }
                assert_eq!(acc.dimensionality, v.dimensionality);
                let fvec = Fvec::from(v);
                acc.data.extend(fvec.data.iter());
                acc
            },
        )
    }
}

/// Dataset sourced from a .fvecs file
pub struct FvecsDataset {
    mmap: Mmap,
    dimensionality: u32,
    index: Option<Box<dyn HnswIndex>>,
}

impl Dataset for FvecsDataset {
    fn new(fname: String) -> Result<Self> {
        let mut pathbuf = PathBuf::new();
        pathbuf.push(fname);

        let f = File::open(pathbuf)?;

        // SAFETY: For the purposes of our benchmarking suite, we are assuming that the underlying
        // file will not be modified throughout the duration of the program, as we control the file
        // system.
        let mmap = unsafe { Mmap::map(&f)? };

        // In OAK, we are assuming that our datasets are always in-memory for the first set of
        // experiments.

        // Calls syscall mlock on file memory, ensuring that it will be in RAM until unlocked.
        // This will through an error if RAM is not large enough.
        let _ = mmap.lock()?;

        // let dimensionality = LittleEndian::read_32(&mmap[..4]);
        let dimensionality = LittleEndian::read_u32(&mmap[..4]);

        Ok(Self {
            index: None,
            mmap,
            dimensionality,
        })
    }

    fn dataset_info(&self) -> String {
        "Section of the Deep1B dataset X vectors".to_string()
    }

    fn get_dimensionality(&self) -> u32 {
        self.dimensionality
    }

    fn get_data(&self) -> Result<Vec<Fvec>> {
        let view = FvecsView::new(&self.mmap);
        let vecs = view.map(|v| Fvec::from(v)).collect();
        Ok(vecs)
    }
}

#[derive(Error, Debug)]
pub enum SearchableError {
    #[error("You must index a dataset before it can be searched")]
    DatasetIsNotIndexed,
}

pub trait HnswIndex {
    fn add(&mut self, vecs: FvecsView);
}

struct FaissHnswIndex {
    index: IndexImpl,
}

// impl FaissHnswIndex {
//     fn build(
//         dimensionality: u32,
//         index_type: VectorIndex,
//         metric_type: VectorMetric,
//     ) -> Result<Self> {
//         let index = index_factory(
//             dimensionality,
//             index_type.to_string(),
//             match metric_type {
//                 VectorMetric::IndexFlatL2 => MetricType::L2,
//             },
//         )?;
//         Ok(Self { index })
//     }
// }

impl HnswIndex for FaissHnswIndex {
    fn add(&mut self, vecs: FvecsView) {
        let flattened_vecs = FlattenedVecs::from(vecs);
        self.index.add(&flattened_vecs.data[..]);
    }
}

pub struct AcornHnswOptions {
    pub m: i32,     // degree bound for traversed nodes during ACORN search
    pub gamma: i32, // neighbor expansion factor for ACORN index
    pub m_beta: i32, // compression parameter for ACORN index
                    // TODO: metadata std::vector<int>&
}

pub struct AcornHnswIndex {
    index: cxx::UniquePtr<ffi::IndexACORNFlat>,
}

#[cfg(feature = "hnsw_faiss")]
impl AcornHnswIndex {
    pub fn new(dataset: &FvecsDataset, options: &AcornHnswOptions) -> Result<Self> {
        let dimensionality = i32::try_from(dataset.get_dimensionality())
            .expect("dimensionality should not be greater than 2,147,483,647");

        let index = ffi::new_index_acorn(dimensionality, options.m, options.gamma, options.m_beta);

        println!("We got an opaque pointer to the thing.");

        // index.add(dataset.len(), dataset.get_raw_ptr());

        Ok(Self { index })
    }
}

#[cfg(feature = "hnsw_rust")]
impl Searchable for FvecsDataset {
    fn build_index(&mut self, index_type: VectorIndex) -> Result<()> {
        Ok(())
    }

    fn search_with_index(
        &self,
        query_vectors: Vec<Fvec>,
        topk: Option<usize>,
    ) -> Result<Vec<Vec<usize>>, SearchableError> {
        Ok(vec![])
    }
}
