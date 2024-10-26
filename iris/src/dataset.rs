// extern crate libc;
use crate::architecture::HardwareArchitecture::{
    self, DramBalancedHnswPartitioned, DramBalancedLshPartitioned, DramRandomPartitioning,
    SsdStandalone,
};
use crate::stubs::gen_random_vecs;
// use std::ffi::c_void;
use std::fs::File;
use std::path::PathBuf;
use std::slice;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use memmap2::Mmap;
use std::marker::PhantomData;
use thiserror::Error;

const FOUR_BYTES: usize = std::mem::size_of::<f32>();

/// Trait for a dataset of vectors.
/// Note that this must be `Sized` in order that the constructor can return a Result.
pub trait Dataset: Sized {
    /// Create a new dataset, loading into memory or keeping on disk as per `hw_arch`.
    fn new(hw_arch: HardwareArchitecture, cluster_size: usize, node_num: usize) -> Result<Self>;
    /// Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    /// Provide the dimensionality of the vectors in the dataset.
    fn get_dimensionality(&self) -> u32;
    /// Returns None if the data is not yet trained, else the HardwareArchitecture on which it was
    /// trained.
    fn get_hardware_architecture(&self) -> HardwareArchitecture;
    /// Returns data in dataset. Fails if full dataset doesn't fit in memory.
    fn get_data(&self) -> Result<Vec<Fvec>>;
}

// https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
#[derive(Debug)]
pub enum VectorIndex {
    IndexFlatL2,
}

impl ToString for VectorIndex {
    fn to_string(&self) -> String {
        match self {
            VectorIndex::IndexFlatL2 => "IndexFlatL2".to_string(),
        }
    }
}

pub trait Searchable: Dataset {
    fn build_index(&mut self, index_type: VectorIndex) -> Result<()>;
    /// Takes a Vec<Fvec> and returns a Vec<Vec<usize>>, whereby each inner Vec<usize> is an array
    /// of the indices for the `topk` vectors returned from the result.
    fn search_with_index(
        &self,
        query_vectors: Vec<Fvec>,
        topk: Option<usize>,
    ) -> Result<Vec<Vec<usize>>, SearchableError>;
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
    dimensionality: u32,
    index: Option<IndexImpl>,
    hw_arch: HardwareArchitecture,
}

impl Dataset for Deep1X {
    fn new(hw_arch: HardwareArchitecture, cluster_size: usize, node_num: usize) -> Result<Self> {
        // TODO: ensure filename formula maps to the output of Python provisioning
        // using `architecture`.
        let mut fname = PathBuf::new();
        // {architecture}_{clustersize}nodes_node{nodenumber}.fvecs.
        let filename = format!("{hw_arch}_{cluster_size}nodes_node{node_num}.fvecs");
        fname.push(filename);

        let f = File::open(fname)?;

        // SAFETY: For the purposes of our benchmarking suite, we are assuming that the underlying
        // file will not be modified throughout the duration of the program, as we control the file
        // system.
        let mmap = unsafe { Mmap::map(&f)? };

        if let DramBalancedHnswPartitioned | DramBalancedLshPartitioned | DramRandomPartitioning =
            hw_arch
        {
            // Calls syscall mlock on file memory, ensuring that it will be in RAM until unlocked.
            // This will through an error if RAM is not large enough.
            let _ = mmap.lock()?;
        }

        // let dimensionality = LittleEndian::read_32(&mmap[..4]);
        let dimensionality = LittleEndian::read_u32(&mmap[..4]);

        // SAFETY:
        // let dimensionality = unsafe {
        //     let value = *(mmap as *const u32);
        //     value
        // };

        Ok(Self {
            index: None,
            mmap,
            hw_arch,
            dimensionality,
        })
    }

    fn dataset_info(&self) -> String {
        "Section of the Deep1B dataset X vectors".to_string()
    }

    fn get_dimensionality(&self) -> u32 {
        self.dimensionality
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

#[derive(Error, Debug)]
pub enum SearchableError {
    #[error("You must index a dataset before it can be searched")]
    DatasetIsNotIndexed,
}

impl Searchable for Deep1X {
    fn build_index(&mut self, index_type: VectorIndex) -> Result<()> {
        let idx = index_factory(
            self.get_dimensionality(),
            index_type.to_string(),
            MetricType::L2,
        )?;
        self.index = Some(idx);

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
        Ok(())
    }

    fn search_with_index(
        &self,
        query_vectors: Vec<Fvec>,
        topk: Option<usize>,
    ) -> Result<Vec<Vec<usize>>, SearchableError> {
        // _, ids = index.search(x=xq, k=topk)
        if self.index.is_none() {
            return Err(SearchableError::DatasetIsNotIndexed);
        }
        Ok(vec![])
    }
}
