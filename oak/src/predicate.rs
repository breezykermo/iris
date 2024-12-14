use crate::dataset::{SearchableError, SimilaritySearchable};
use crate::fvecs::FvecsDataset;
use anyhow::Result;

/// ACORN specifies the predicates for queries as one bitmap per query, where the bitmap is an
/// array of length N (the number of total entries in the database). This is presumably so that
/// ACORN is capable of arbitrary predicates; but it does mean that the query language for the cpp
/// API is somewhat unwieldy.
///
/// Thus here we allow a more concise representation of predicates with a basic query language that
/// we can develop as needed. For the POC, the query language _only_ accepts equality for a
/// singular match. The assumption is also that each vector has one and only one attribute (a u8).
/// At a first order of approximation, this language accepts constructs with two tokens,
/// PredicateOp, and PredicateRhs. A basic example: to express the idea that we only want to
/// retrieve queries "where the attribute matches '10'", we would construct a query as follows:
///
/// query = PredicateQuery {
///     op: PredicateOp::Equals,
///     rhs: PredicateRhs::Number(10),
/// }
///
/// Only u8 numbers are supported for names and predicate values at present, as the assumption is
/// that a dataset will be loaded in with a CSV with columns whose titles are all u8 (the `lhs` of
/// a PredicateQuery), and whose values are _also_ all u8s (the `rhs`).
///
/// If necessary, we can expand this later.
#[derive(Clone)]
pub struct PredicateQuery {
    pub op: PredicateOp,
    pub rhs: PredicateRhs,
}

#[derive(Clone)]
pub enum PredicateOp {
    Equals,
}

#[derive(Clone)]
pub enum PredicateRhs {
    Number(u8),
}

impl From<&PredicateRhs> for i32 {
    fn from(value: &PredicateRhs) -> Self {
        match value {
            PredicateRhs::Number(num) => *num as i32,
        }
    }
}

impl PredicateQuery {
    /// Creates a new PredicateQuery where the attribute is equal to the provided argument `num`.
    pub fn new(num: u8) -> Self {
        Self {
            op: PredicateOp::Equals,
            rhs: PredicateRhs::Number(num),
        }
    }
}
