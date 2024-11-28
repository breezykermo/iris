use crate::fvecs::{FlattenedVecs, Fvec};
use crate::predicate::PredicateQuery;
// use tracing::info;

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

/// These parameters are currently essentially ACORN parameters, taken from https://github.com/csirianni/ACORN/blob/main/README.md
pub struct OakIndexOptions {
    /// Degree bound for traversed nodes during ACORN search
    pub m: i32,
    /// Neighbor expansion factor for ACORN index
    pub gamma: i32,
    /// Compression parameter for ACORN index
    pub m_beta: i32,
}

/// Trait for a dataset of vectors.
/// Note that this must be `Sized` in order that the constructor can return a Result.
pub trait Dataset: Sized {
    /// Create a new dataset, loading all fvecs into memory. The `fname` should represent a
    /// filename that corresponds to both a "{fname}.fvecs" that contains the vectors, and a
    /// "{fname}.csv" that contains the attributes (over which predicates can be constructed) for
    /// those vectors. Each row in the CSV corresponds to the vector at the same index in the fvecs
    /// file, and each column represents an attribute on that vector.
    fn new(fname: String) -> Result<Self>;
    /// Initialize the index with the vectors from the dataset.
    fn initialize(&mut self, opts: &OakIndexOptions) -> Result<(), ConstructionError>;
    /// Provide the number of vectors that have been added to the dataset.
    fn len(&self) -> usize;
    /// Returns a vec of u8s where a 1 represents that the specified attribute exists on the vector
    /// at the given index in the database. If the attribute doesn't exist in the database, an
    /// appropriate error is returned. The vec will be of length `self.len()`.
    fn attribute_equals_map(&self, attribute: u8) -> Result<bool>;
    /// Provide the dimensionality of the vectors in the dataset.
    fn get_dimensionality(&self) -> usize;
    /// Returns data in dataset. Fails if full dataset doesn't fit in memory.
    fn get_data(&self) -> Result<Vec<Fvec>>;
    /// Takes a Vec<Fvec> and returns a Vec<Vec<(usize, f32)>>, whereby each inner Vec<(usize, f32)> is an array
    /// of tuples in which t[0] is the index of the resthe `topk` vectors returned from the result.
    fn search(
        &self,
        query_vectors: FlattenedVecs,
        predicate_query: Option<PredicateQuery>,
        topk: usize,
    ) -> Result<Vec<TopKSearchResult>, SearchableError>;
}
