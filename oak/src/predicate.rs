use anyhow::Result;
use core::ffi::c_char;
use thiserror::Error;

/// The errors that can be returned from serializing a query over a dataest.
#[derive(Error, Debug)]
pub enum PredicateSerializationError {
    #[error("The attribute specified in the `lhs` of this PredicateQuery does not exist in the dataset.")]
    NoSuchAttributeExists,
}

/// ACORN specifies the predicates for queries as one bitmap per query, where the bitmap is an
/// array of length N (the number of total entries in the database). This is presumably so that
/// ACORN is capable of arbitrary predicates; but it does mean that the query language for the cpp
/// API is somewhat unwieldy.
///
/// Thus here we allow a more concise representation of predicates with a basic query language that
/// we can develop as needed. For the POC, the query language _only_ accepts equality for a
/// singular match. At a first order of approximation, this language accepts constructs with three
/// tokens, PredicateLhs, PredicateOp, and PredicateRhs. A basic example: to express the idea that
/// we only want to retrieve queries "where attribute with the name '1' matches '10'", we would
/// construct a query as follows:
///
/// query = PredicateQuery {
///     lhs: PredicateLhs::Number(1),
///     op: PredicateOp::Equals,
///     rhs: PredicateRhs::Number(10),
/// }
///
/// Only u8 numbers are supported for names and predicate values at present, as the assumption is
/// that a dataset will be loaded in with a CSV with columns whose titles are all u8 (the `lhs` of
/// a PredicateQuery), and whose values are _also_ all u8s (the `rhs`).
///
/// If necessary, we can expand this later.
pub struct PredicateQuery {
    lhs: PredicateLhs,
    op: PredicateOp,
    rhs: PredicateRhs,
}

pub enum PredicateLhs {
    Number(u8),
}

pub enum PredicateOp {
    Equals,
}

pub enum PredicateRhs {
    Number(u8),
}

impl PredicateQuery {
    /// 'Serializes' a query as a filter map, which will allow it to be passed across FFI to the
    /// `search` function. A filter map is specific to a dataset, as it is of length (nq * N),
    /// where nq is the number of queries in the map, and N is the number of vectors in the
    /// dataset. A value of 1 in the bitmap represents that the search query matches with the
    /// vector at that index in the dataset, and a value of 0 that it doesn't.
    ///
    /// If an attribute matching the `lhs` of the query does not exist in the dataset, then an
    /// error will be raised.
    pub fn serialize_as_filter_map(
        &self,
        // dataset: &Box<dyn Dataset>,
    ) -> Result<Vec<c_char>, PredicateSerializationError> {
        unimplemented!();
        // let mut filter_id_map: Vec<c_char> = vec![0; number_of_query_vectors * count];
        //
        // for xq in 0..number_of_query_vectors {
        //     for xb in 0..count {
        //         if metadata[xb] == aq[xq] {
        //             filter_id_map[xq * count + xb] = 1;
        //         }
        //     }
        // }
        //
        // filter_id_map
    }
}
