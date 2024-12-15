use crate::acorn::AcornHnswIndex;
use crate::bitmask::Bitmask;
use crate::dataset::{
    ConstructionError, HybridSearchMetadata, OakIndexOptions, SearchableError,
    SimilaritySearchable, TopKSearchResult,
};
use crate::predicate::PredicateQuery;
use slog_scope::debug;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use csv::ReaderBuilder;
use memmap2::Mmap;
// use slog_scope::debug;
use std::fs::File;
use std::path::PathBuf;

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

#[allow(dead_code)]
pub struct Fvec {
    dimensionality: usize,
    data: Vec<f32>,
}

pub struct FlattenedVecs {
    // The length of each vector in the flattened representation.
    pub dimensionality: usize,
    // This data is the flattened representation of all vectors of size `dimensionality`.
    pub data: Vec<f32>,
}

impl FlattenedVecs {
    pub fn len(&self) -> usize {
        self.data.len() / self.dimensionality
    }

    pub fn to_vec(self) -> Vec<FlattenedVecs> {
        self.data
            .chunks(self.dimensionality)
            .map(|c| FlattenedVecs {
                dimensionality: self.dimensionality,
                data: c.to_vec(),
            })
            .collect()
    }

    /// Creates a new FlattenedVecs based on a bitmask and an original one.
    /// Only the necessary items (items that match the bitmask) are copied.
    pub fn clone_via_bitmask(&self, bitmask: &Bitmask) -> Self {
        let new_data: Vec<f32> = self
            .data
            .chunks_exact(self.dimensionality)
            .zip(bitmask.map.iter())
            .filter_map(|(vector, &keep)| {
                if keep == 1 {
                    Some(vector) // Keep the vector if bitmask says so
                } else {
                    None
                }
            })
            .flat_map(|vector| vector.iter().copied())
            .collect();

        Self {
            dimensionality: self.dimensionality,
            data: new_data,
        }
    }

    /// Create a 'flattened' representation of fvecs (meaning that the vectors are simply contiguous to
    /// each other in memory, rather than prepended by their dimensionality explicitly as in the .fvecs
    /// representation). This is necessary reformatting for calling ACORN methods via FFI, and the
    /// transformation also ensures that the vectors are in memory (rather than on disk).
    pub fn read_from_mmap(mmap: &Mmap, count: usize, dimensionality: usize) -> Self {
        let mut all_fvecs = Vec::with_capacity(count * dimensionality);
        let fvec_len_in_bytes = (dimensionality + 1) * FOUR_BYTES;
        let mut current_index = 0;
        for i in mmap.chunks_exact(fvec_len_in_bytes) {
            let mut fvecs = parse_u8_to_f32(&i);
            // skip the dimensionality, we don't need it in the flattened.
            fvecs.drain(0..1);
            all_fvecs.splice(current_index..current_index, fvecs);
            current_index += dimensionality;
        }

        Self {
            dimensionality,
            data: all_fvecs,
        }
    }
}

impl From<&FvecsDataset> for FlattenedVecs {
    fn from(dataset: &FvecsDataset) -> Self {
        FlattenedVecs::read_from_mmap(&dataset.mmap, dataset.count, dataset.dimensionality)
    }
}

/// Create a Vec<PredicateQuery> representing queries that match the specified attribute in the
/// query vectors. The 0th element in the returned Vec, for example, will be a PredicateQuery for
/// all vectors that match attribute X, where X is the attribute on the 0th query vector.
impl From<&FvecsDataset> for Vec<PredicateQuery> {
    fn from(dataset: &FvecsDataset) -> Self {
        dataset
            .metadata
            .as_ref()
            .iter()
            // NOTE: we assume here that the attribute loaded is safe to cast to a u8, as we have
            // generated the attributes as such. The reason that `dataset.metadata` is a u32 is
            // because this is what the ACORN code expects (and as such should probably be changed
            // to a u8 so that implementation details aren't leaky to higher-level abstractions).
            .map(|x| PredicateQuery::new((*x).try_into().unwrap()))
            .collect()
    }
}

/// Dataset sourced from a .fvecs file
pub struct FvecsDataset {
    pub mmap: Mmap,
    count: usize,
    dimensionality: usize,
    index: Option<AcornHnswIndex>,
    pub metadata: HybridSearchMetadata,
    pub flat: FlattenedVecs,
}

impl FvecsDataset {
    /// Create a new dataset, loading all fvecs into memory. The `fname` should represent a
    /// filename that corresponds to both a "{fname}.fvecs" that contains the vectors, and a
    /// "{fname}.csv" that contains the attributes (over which predicates can be constructed) for
    /// those vectors. Each row in the CSV corresponds to the vector at the same index in the fvecs
    /// file, and each column represents an attribute on that vector.
    pub fn new(fname: String, load_csv: bool) -> Result<Self> {
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
        // let _ = mmap.lock()?;

        let dimensionality = LittleEndian::read_u32(&mmap[..FOUR_BYTES]) as usize;
        assert_eq!(
            dimensionality,
            LittleEndian::read_u32(&mmap[0..FOUR_BYTES]) as usize
        );

        // Each fvec is a dimensionality (4 bytes) followed by `dimensionality` number of f32
        // values. Fvecs are contiguous in the file.
        let count = &mmap[..].len() / ((1 + dimensionality) * FOUR_BYTES);
        debug!(
            "First dimensionality read from file is {dimensionality}; assuming the same for all remaining."
        );
        debug!("The file read has {count} vectors.");

        let metadata = if load_csv {
            let mut metadata_fname = PathBuf::new();
            metadata_fname.push(&format!("{}.csv", fname));
            let metadata_vec = read_csv_to_vec(&metadata_fname)?;
            HybridSearchMetadata::new(metadata_vec)
        } else {
            // NOTE: this is a bad hack. It should really be an option
            HybridSearchMetadata::new(vec![])
        };

        let flat = FlattenedVecs::read_from_mmap(&mmap, count, dimensionality);
        Ok(Self {
            index: None,
            count,
            mmap,
            dimensionality,
            metadata,
            flat,
        })
    }

    #[allow(dead_code)]
    fn get_data(&self) -> Result<Vec<Fvec>> {
        let vecs = self
            .flat
            .data
            .chunks_exact(self.dimensionality)
            .map(|x| Fvec {
                dimensionality: self.dimensionality,
                data: x.to_vec(),
            })
            .collect();
        Ok(vecs)
    }

    pub fn view(&self, pq: &PredicateQuery) -> FvecsDatasetPartition {
        let mask = Bitmask::new(pq, self);
        let metadata = HybridSearchMetadata::new_from_bitmask(&self.metadata, &mask);

        FvecsDatasetPartition {
            base: self,
            mask,
            flat: None,
            index: None,
            metadata,
        }
    }
}

impl SimilaritySearchable for FvecsDataset {
    fn len(&self) -> usize {
        self.count
    }

    fn get_dimensionality(&self) -> usize {
        self.dimensionality as usize
    }

    fn get_metadata(&self) -> &HybridSearchMetadata {
        &self.metadata
    }

    fn initialize(&mut self, opts: &OakIndexOptions) -> Result<(), ConstructionError> {
        let index = AcornHnswIndex::new(self, &self.flat, opts)?;
        self.index = Some(index);
        Ok(())
    }

    fn search(
        &self,
        query_vectors: &FlattenedVecs,
        predicate_query: &Option<PredicateQuery>,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        if self.index.is_none() {
            return Err(SearchableError::DatasetIsNotIndexed);
        }

        debug!("query_vectors len: {}", query_vectors.len());
        debug!("fvecs dataset len: {}", self.len());

        let mut mask = match predicate_query {
            None => Bitmask::new_full(self),
            Some(pq) => Bitmask::new(pq, self),
        };

        self.index
            .as_ref()
            .unwrap()
            .search(query_vectors, &mut mask.map, topk, efsearch)
    }

    fn search_with_bitmask(
        &self,
        query_vectors: &FlattenedVecs,
        bitmask: &Bitmask,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        let mut filter_id_map = Vec::<i8>::from(bitmask);

        // TODO: this & to filter_id_map should not have to be mutable
        self.index
            .as_ref()
            .unwrap()
            .search(query_vectors, &mut filter_id_map, topk, efsearch)
    }
}

/// A 'partition' of the FvecsDataset, originally represented just by a base dataset and a Bitmask.
pub struct FvecsDatasetPartition<'a> {
    base: &'a FvecsDataset,
    mask: Bitmask,
    index: Option<AcornHnswIndex>,
    /// We have an Option here so that the copying of the base vectors can be deferred to the point
    /// at which we decide to build the index. This is an implementation detail, as one could
    /// imagine a pure Rust implementation of the search methods that does not require this
    /// original copy.
    flat: Option<FlattenedVecs>,
    /// The same with the metadata
    metadata: HybridSearchMetadata,
}

impl<'a> SimilaritySearchable for FvecsDatasetPartition<'a> {
    fn len(&self) -> usize {
        self.mask.bitcount()
    }

    fn get_metadata(&self) -> &HybridSearchMetadata {
        &self.metadata
    }

    fn get_dimensionality(&self) -> usize {
        self.base.dimensionality
    }

    fn initialize(&mut self, opts: &OakIndexOptions) -> Result<(), ConstructionError> {
        let og = &self.base.flat;
        let flat = og.clone_via_bitmask(&self.mask);

        let index = AcornHnswIndex::new(self, &flat, opts)?;

        self.index = Some(index);
        self.flat = Some(flat);

        Ok(())
    }

    fn search(
        &self,
        query_vectors: &FlattenedVecs,
        predicate_query: &Option<PredicateQuery>,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        if self.index.is_none() {
            return Err(SearchableError::DatasetIsNotIndexed);
        }

        debug!("query_vectors len: {}", query_vectors.len());
        debug!("fvecs dataset len: {}", self.len());

        let mut mask = match predicate_query {
            None => Bitmask::new_full(self),
            Some(pq) => Bitmask::new(pq, self),
        };

        self.index
            .as_ref()
            .unwrap()
            .search(query_vectors, &mut mask.map, topk, efsearch)
    }

    fn search_with_bitmask(
        &self,
        query_vectors: &FlattenedVecs,
        bitmask: &Bitmask,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        let mut filter_id_map = Vec::<i8>::from(bitmask);

        // TODO: this & to filter_id_map should not have to be mutable
        self.index
            .as_ref()
            .unwrap()
            .search(query_vectors, &mut filter_id_map, topk, efsearch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::generate_random_vector;
    use std::sync::Once;

    #[test]
    fn test_not_initialized_error() {
        let dataset = FvecsDataset::new("data/sift_query".to_string(), false).unwrap();
        let predicate: Option<PredicateQuery> = None;
        let dimensionality = dataset.dimensionality;
        let query_vector = FlattenedVecs {
            dimensionality,
            data: generate_random_vector(dimensionality),
        };

        assert!(dataset.index.is_none());
        // TODO: the following gives a build error; something to do with
        // undefined reference to `typeinfo for faiss::FaissException'
        // i.e. the handling of errors across the FFI boundary.
        // let result = dataset.search(query_vector, predicate, 1);
        // assert_eq!(result, Err(SearchableError::DatasetIsNotIndexed));
    }

    #[test]
    fn test_fvecs_to_flattened_vec() {
        let dataset = FvecsDataset::new("data/sift_query".to_string(), true).unwrap();
        let dataset_len = dataset.len();
        let vecs = FlattenedVecs::from(&dataset);

        assert_eq!(vecs.len(), dataset_len);
    }
}

