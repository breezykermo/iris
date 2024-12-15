use crate::bitmask::Bitmask;
use crate::fvecs::FlattenedVecs;
use crate::predicate::PredicateQuery;

use anyhow::Result;
use thiserror::Error;

/// The errors that can be returned from searching an OAK dataset.
#[derive(Error, Debug, PartialEq)]
pub enum SearchableError {
    #[error("You must index a dataset before it can be searched")]
    DatasetIsNotIndexed,
    #[error("Could not serialize the predicate")]
    PredicateSerializationError,
    #[error("Underlying C++ error: {0}")]
    CppError(String),
}

impl From<cxx::Exception> for SearchableError {
    fn from(err: cxx::Exception) -> Self {
        // Customize the conversion logic as needed
        SearchableError::CppError(err.to_string())
    }
}

/// The errors that can be returned from constructing an OAK dataset.
#[derive(Error, Debug)]
pub enum ConstructionError {}

/// t[0] is the index of the vector that is similar in the dataset, t[1] is a f32 representing the
/// distance of the found vector from the original query.
pub type SimilaritySearchResult = (usize, f32);

// A vec of length `k` with tuples representing the similarity search results.
pub type TopKSearchResult = Vec<SimilaritySearchResult>;

// A batch of items with type `TopKSearchResult`.
pub type TopKSearchResultBatch = Vec<TopKSearchResult>;

/// The type in which the attributes for hybrid search are notated. At the moment the assumed
/// constraint is that there is at most one attribute per vector, and it is always an i32.
pub struct HybridSearchMetadata {
    attrs: Vec<i32>,
}

impl HybridSearchMetadata {
    pub fn new(attrs: Vec<i32>) -> Self {
        Self { attrs }
    }

    pub fn new_from_bitmask(other: &Self, mask: &Bitmask) -> Self {
        let filtered_attrs: Vec<i32> = other
            .attrs
            .iter()
            .zip(mask.map.iter())
            .filter_map(|(&attr, &keep)| {
                if keep == 1 {
                    Some(attr) // Keep the attribute if the bitmask allows
                } else {
                    None
                }
            })
            .collect();

        HybridSearchMetadata {
            attrs: filtered_attrs,
        }
    }

    pub fn len(&self) -> usize {
        self.attrs.len()
    }
}

impl AsRef<Vec<i32>> for HybridSearchMetadata {
    fn as_ref(&self) -> &Vec<i32> {
        self.attrs.as_ref()
    }
}

/// These parameters are currently essentially ACORN parameters, taken from
/// https://github.com/csirianni/ACORN/blob/main/README.md
pub struct OakIndexOptions {
    /// Degree bound for traversed nodes during ACORN search
    pub m: i32,
    /// Neighbor expansion factor for ACORN index
    pub gamma: i32,
    /// Compression parameter for ACORN index
    pub m_beta: i32,
}

/// The default options for OAK are the options suggested in the ACORN readme: https://github.com/csirianni/ACORN/blob/main/README.md
impl Default for OakIndexOptions {
    fn default() -> Self {
        Self {
            gamma: 1,
            m: 32,
            m_beta: 64,
        }
    }
}

/// Trait for a dataset of vectors.

pub trait SimilaritySearchable {
    /// Provide the number of vectors that have been added to the dataset.
    fn len(&self) -> usize;

    /// Provide the dimensionality of the vectors in the dataset.
    fn get_dimensionality(&self) -> usize;

    /// Get the metadata that represents the attributes over the vectors (for hybrid search).
    fn get_metadata(&self) -> &HybridSearchMetadata;

    /// Build the index associated with this dataset. If an index has not been built, all search
    /// methods will throw an error.
    fn initialize(&mut self, opts: &OakIndexOptions) -> Result<(), ConstructionError>;

    /// Takes a Vec<Fvec> and returns a Vec<Vec<(usize, f32)>>, whereby each inner Vec<(usize, f32)> is an array
    /// of tuples in which t[0] is the index of the resthe `topk` vectors returned from the result.
    fn search(
        &self,
        query_vectors: &FlattenedVecs,
        predicate_query: &Option<PredicateQuery>,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError>;

    /// Takes a Vec<Fvec> and returns a Vec<Vec<(usize, f32)>>, whereby each inner Vec<(usize, f32)> is an array
    /// of tuples in which t[0] is the index of the resthe `topk` vectors returned from the result.
    fn search_with_bitmask(
        &self,
        query_vectors: &FlattenedVecs,
        bitmask: &Bitmask,
        topk: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError>;
}
