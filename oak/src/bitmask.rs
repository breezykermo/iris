use crate::dataset::Dataset;
use crate::predicate::{PredicateOp, PredicateQuery};
use core::ffi::c_char;

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
    pub fn new<D: Dataset>(pq: &PredicateQuery, dataset: &D) -> Self {
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

    pub fn new_full<D: Dataset>(dataset: &D) -> Self {
        let map = vec![true as i8; dataset.len()];
        let bitcount = 1;
        Self { map, bitcount }
    }

    pub fn bitcount(&self) -> usize {
        self.bitcount
    }
}

impl From<Bitmask> for Vec<i8> {
    fn from(mask: Bitmask) -> Self {
        mask.map
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
