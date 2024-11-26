use std::time::Duration;

use crate::dataset::FlattenedVecs;
/**
 * Queries are FVecs file and we specify some k for a k-nearest neighbors search
 * For acorn, the query sets must first filter by some predicate and then do the 
 * k-NN search. 
 * 
 * Workloads:
 * 1) All queries filter on the same predicate -> index pre-made
 * 2) Queries filter on many different predicates -> we only have space for some
 * indices and the rest must be brute-forced
 * 3) Creating an index in the middle?
 */
// Trait for different loads
pub trait Load {
    fn load_info(&self) -> String;
}

// SyncQueries load implementation
pub struct SyncQueries {
    pub num_queries: usize, // e.g., 10k queries
}

pub struct ConstPacedQueries {
    pub num_queries: usize,
    pub filter_id_map: Vec<c_char>,
    pub k: usize,
    pub query_vectors : Vec<FlattenedVecs>,
}
// Is this to load the query set into memory? Vec<FVEC>?
impl Load for SyncQueries {
    fn load_info(&self) -> String {
        format!("{} synchronous queries", self.num_queries)
    }
}
/**
 * This was taken from - https://github.com/akshayknarayan/burrito/blob/1649718d9acb96c440ee423cf43688ffbeed9a5d/kvstore-ycsb/src/lib.rs#L73 -
 * a benchmarking project. It was modified to fit vector operations instead
 * of DB operations
 */

pub fn const_paced_ops_stream(
    ops: Vec<FlattenedVecs>,
    filter_id_map: &mut Vec<c_char>,
    k: usize,
    interarrival_micros: u64,
    // client_id: usize,
) -> impl Stream<Item = (usize, (FlattenedVecs, Vec<c_char>, usize))> {
    let len = ops.len();
    let tkr = poisson_ticker::SpinTicker::new_const_with_log_id( // Can't track this down but some kind of library func?
        Duration::from_micros(interarrival_micros),
        client_id,
    )
    .zip(futures_util::stream::iter((0..len).rev()));
    // tkr.zip(futures_util::stream::iter(ops))
    tkr.zip(futures_util::stream::iter(ops))
    .map(move |((_, i), vec)| {
        // Each item in the stream will include:
        // (Index, Operation, Filter Map, K-value))
        (i, (o, filter_id_map.clone(), k))
    })
}