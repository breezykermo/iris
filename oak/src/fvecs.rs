use crate::acorn::AcornHnswIndex;
use crate::dataset::{
    ConstructionError, Dataset, OakIndexOptions, SearchableError, TopKSearchResult,
};
use crate::predicate::PredicateQuery;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use csv::ReaderBuilder;
use memmap2::Mmap;
use slog_scope::{debug, info};
use std::fs::File;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::slice;

const FOUR_BYTES: usize = std::mem::size_of::<f32>();

fn read_csv_to_vec(file_path: &PathBuf) -> Result<Vec<i32>> {
    // Open the file
    let file = File::open(file_path)?;

    // Create a CSV reader
    let mut reader = ReaderBuilder::new()
        .has_headers(false) // No headers in this example
        .from_reader(file);

    // Collect integers from the CSV
    let mut numbers = Vec::new();
    for result in reader.records() {
        let record = result?;
        if let Some(field) = record.get(0) {
            // Assuming a single column
            numbers.push(field.parse::<i32>()?);
        }
    }

    debug!("{} attributes loaded from CSV.", numbers.len());

    Ok(numbers)
}

/// Converts a slice of `u8` into a `Vec<f32>` assuming little-endian format.
///
/// # Panics
/// - Panics if the length of the input slice is not a multiple of 4.
///
/// # Parameters
/// - `data`: A byte slice containing the raw `f32` data.
///
/// # Returns
/// A vector of `f32` values parsed from the byte slice.
pub fn parse_u8_to_f32(data: &[u8]) -> Vec<f32> {
    assert!(
        data.len() % 4 == 0,
        "Input data length must be a multiple of 4"
    );

    let mut result = Vec::with_capacity(data.len() / FOUR_BYTES);

    for chunk in data.chunks_exact(FOUR_BYTES) {
        let value = LittleEndian::read_f32(chunk);
        result.push(value);
    }

    result
}

pub struct Fvec {
    dimensionality: usize,
    data: Vec<f32>,
}

pub struct FlattenedVecs {
    pub dimensionality: usize,
    // This data is the flattened representation of all vectors of size `dimensionality`.
    pub data: Vec<f32>,
}

impl FlattenedVecs {
    pub fn len(&self) -> usize {
        self.data.len() / self.dimensionality
    }
}

impl From<&FvecsDataset> for FlattenedVecs {
    fn from(dataset: &FvecsDataset) -> Self {
        let mut all_fvecs = Vec::with_capacity(dataset.count * dataset.dimensionality);

        let fvec_len_in_bytes = (dataset.dimensionality + 1) * FOUR_BYTES;
        let mut current_index = 0;
        for i in dataset.mmap.chunks_exact(fvec_len_in_bytes) {
            let mut fvecs = parse_u8_to_f32(&i);
            // skip the dimensionality, we don't need it in the flattened.
            fvecs.drain(0..1);
            all_fvecs.splice(current_index..current_index, fvecs);
            current_index += dataset.dimensionality;
        }

        FlattenedVecs {
            dimensionality: dataset.dimensionality,
            data: all_fvecs,
        }
    }
}
/// Dataset sourced from a .fvecs file
pub struct FvecsDataset {
    pub mmap: Mmap,
    count: usize,
    dimensionality: usize,
    index: Option<Box<AcornHnswIndex>>,
    pub metadata: Vec<i32>,
}

impl Dataset for FvecsDataset {
    fn new(fname: String) -> Result<Self> {
        let mut fvecs_fname = PathBuf::new();
        fvecs_fname.push(&format!("{}.fvecs", fname));

        let f = File::open(fvecs_fname)?;

        // SAFETY: For the purposes of our benchmarking suite, we are assuming that the underlying
        // file will not be modified throughout the duration of the program, as we control the file
        // system.
        let mmap = unsafe { Mmap::map(&f)? };

        // In OAK, we are assuming that our datasets are always in-memory for the first set of
        // experiments.

        // Calls syscall mlock on file memory, ensuring that it will be in RAM until unlocked.
        // This will throw an error if RAM is not large enough.
        let _ = mmap.lock()?;

        let dimensionality = LittleEndian::read_u32(&mmap[..FOUR_BYTES]) as usize;
        assert_eq!(
            dimensionality,
            LittleEndian::read_u32(&mmap[0..FOUR_BYTES]) as usize
        );

        // Each fvec is a dimensionality (4 bytes) followed by `dimensionality` number of f32
        // values. Fvecs are contiguous in the file.
        let count = &mmap[..].len() / ((1 + dimensionality) * FOUR_BYTES);
        debug!(
            "Firest dimensionality read from file is {dimensionality}; assuming the same for all remaining."
        );
        debug!("The file read has {count} vectors.");

        let mut metadata_fname = PathBuf::new();
        metadata_fname.push(&format!("{}.csv", fname));

        let metadata = read_csv_to_vec(&metadata_fname)?;

        Ok(Self {
            index: None,
            count,
            mmap,
            dimensionality,
            metadata,
        })
    }

    fn initialize(&mut self, opts: &OakIndexOptions) -> Result<(), ConstructionError> {
        let main_index = AcornHnswIndex::new(&self, opts)?;
        self.index = Some(Box::new(main_index));
        Ok(())
    }

    fn len(&self) -> usize {
        self.count
    }

    fn attribute_equals_map(&self, attribute: u8) -> Result<bool> {
        unimplemented!()
    }

    fn get_dimensionality(&self) -> usize {
        self.dimensionality as usize
    }

    fn get_data(&self) -> Result<Vec<Fvec>> {
        let flattened = FlattenedVecs::from(self);
        let vecs = flattened
            .data
            .chunks_exact(self.dimensionality)
            .map(|x| Fvec {
                dimensionality: self.dimensionality,
                data: x.to_vec(),
            })
            .collect();
        Ok(vecs)
    }

    fn search(
        &self,
        query_vectors: FlattenedVecs,
        predicate_query: Option<PredicateQuery>,
        topk: usize,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        if self.index.is_none() {
            return Err(SearchableError::DatasetIsNotIndexed);
        }

        debug!("query_vectors len: {}", query_vectors.len());
        debug!("fvecs dataset len: {}", self.len());

        let mut filter_id_map = match predicate_query {
            None => vec![true as i8; self.len() * query_vectors.len()],
            Some(pq) => pq.serialize_as_filter_map(self)?,
        };

        self.index
            .as_ref()
            .unwrap()
            .search(&query_vectors, &mut filter_id_map, topk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::generate_random_vector;
    use std::sync::Once;

    #[test]
    fn test_not_initialized_error() {
        let dataset = FvecsDataset::new("data/sift_query".to_string()).unwrap();
        let predicate = None;
        let dimensionality = dataset.dimensionality;
        let query_vector = FlattenedVecs {
            dimensionality,
            data: generate_random_vector(dimensionality),
        };

        assert!(dataset.index.is_none());
        let result = dataset.search(query_vector, predicate, 1);
        assert_eq!(result, Err(SearchableError::DatasetIsNotIndexed));
    }

    #[test]
    fn test_fvecs_to_flattened_vec() {
        let dataset = FvecsDataset::new("data/sift_query".to_string()).unwrap();
        let dataset_len = dataset.len();
        let vecs = FlattenedVecs::from(&dataset);

        assert_eq!(vecs.len(), dataset_len);
    }
}
