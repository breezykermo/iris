use crate::dataset::{Dataset, SearchableError};
use crate::fvecs::FvecsDataset;
use anyhow::Result;
use core::ffi::c_char;

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
pub struct PredicateQuery {
    pub op: PredicateOp,
    pub rhs: PredicateRhs,
}

pub enum PredicateOp {
    Equals,
}

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
        dataset: &FvecsDataset,
    ) -> Result<Vec<c_char>, SearchableError> {
        let ds_len = dataset.len();
        let mut filter_id_map: Vec<c_char> = vec![0; ds_len];

        assert_eq!(dataset.metadata.len(), filter_id_map.len());

        let rhs: i32 = i32::from(&self.rhs);

        for (i, xq) in dataset.metadata.iter().enumerate() {
            match self.op {
                PredicateOp::Equals => filter_id_map[i] = (xq == &rhs) as c_char,
            }
        }

        Ok(filter_id_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize() {
        let dataset = FvecsDataset::new("data/sift_query".to_string()).unwrap();
        let pq = PredicateQuery {
            op: PredicateOp::Equals,
            rhs: PredicateRhs::Number(10),
        };

        let bitmap = pq.serialize_as_filter_map(&dataset).unwrap();
        let one = 1 as c_char;
        assert!(bitmap.contains(&one));
    }
}
