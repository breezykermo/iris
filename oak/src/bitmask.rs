use crate::dataset::SimilaritySearchable;
use crate::predicate::{PredicateOp, PredicateQuery};
use core::ffi::c_char;
use std::collections::HashSet;

pub struct Bitmask {
    pub map: Vec<i8>,
    pub bitcount: usize,
}

impl Bitmask {
    /// 'Serializes' a query as a filter map, which will allow it to be passed across FFI to the
    /// `search` function. A filter map is specific to a dataset, as it is of length (nq * N),
    /// where nq is the number of queries in the map, and N is the number of vectors in the
    /// dataset. A value of 1 in the bitmap represents that the search query matches with the
    /// vector at that index in the dataset, and a value of 0 that it doesn't.
    ///
    /// If an attribute matching the `lhs` of the query does not exist in the dataset, then an
    /// error will be raised.
    pub fn new<D: SimilaritySearchable>(pq: &PredicateQuery, dataset: &D) -> Self {
        let ds_len = dataset.len();
        let mut map: Vec<c_char> = vec![0; ds_len];

        assert_eq!(dataset.get_metadata().len(), map.len());

        let rhs: i32 = i32::from(&pq.rhs);
        let on_bit: c_char = 1;
        let mut bitcount: usize = 0;

        for (i, xq) in dataset.get_metadata().as_ref().iter().enumerate() {
            let bit = match pq.op {
                PredicateOp::Equals => (xq == &rhs) as c_char,
            };
            if bit.eq(&on_bit) {
                bitcount += 1;
            }
            map[i] = bit;
        }

        Self { map, bitcount }
    }

    pub fn capacity(&self) -> usize {
        self.map.len()
    }

    pub fn bitcount(&self) -> usize {
        self.bitcount
    }

    pub fn new_full<D: SimilaritySearchable>(dataset: &D) -> Self {
        let map = vec![true as i8; dataset.len()];
        let bitcount = map.len();
        Self { map, bitcount }
    }

    pub fn to_hashset(&self) -> HashSet<i8> {
        self.map.clone().into_iter().collect()
    }

    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        // Convert vectors to sets
        let set1 = self.to_hashset();
        let set2 = other.to_hashset();

        // Calculate the intersection and union
        let intersection: HashSet<_> = set1.intersection(&set2).cloned().collect();
        let union: HashSet<_> = set1.union(&set2).cloned().collect();

        // Calculate Jaccard similarity
        if union.is_empty() {
            0.0 // Handle edge case when both sets are empty
        } else {
            intersection.len() as f64 / union.len() as f64
        }
    }
}

impl From<Bitmask> for Vec<i8> {
    fn from(mask: Bitmask) -> Self {
        mask.map
    }
}

impl From<&Bitmask> for Vec<i8> {
    fn from(mask: &Bitmask) -> Self {
        // NOTE: we take cloning the Bitmask here as acceptable, as the values are only i8s
        mask.map.clone()
    }
}

impl From<Vec<i32>> for Bitmask {
    fn from(attrs: Vec<i32>) -> Bitmask {
        let bitcount = attrs.iter().sum::<i32>() as usize;
        let map = attrs.into_iter().map(|x| x as i8).collect();
        Self { map, bitcount }
    }
}

impl From<Vec<i8>> for Bitmask {
    fn from(map: Vec<i8>) -> Bitmask {
        let bitcount = map.iter().sum::<i8>() as usize;
        Self { map, bitcount }
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

        let bitmap = Bitmask::new(&pq, &dataset);
        let one = 1 as c_char;
        assert!(bitmap.map.contains(&one));
    }
}
