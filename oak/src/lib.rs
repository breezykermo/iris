#[cxx::bridge(namespace = "faiss")]
pub mod ffi {
    unsafe extern "C++" {
        include!("oak/third_party/ACORN/faiss/IndexACORN.h");

        type IndexACORNFlat;

        fn new_index_acorn(
            d: i32,
            M: i32,
            gamma: i32,
            M_beta: i32,
            metadata: &Vec<i32>,
        ) -> UniquePtr<IndexACORNFlat>;

        unsafe fn add_to_index(
            idx: &mut UniquePtr<IndexACORNFlat>,
            n: i64,        // number of vectors to be added
            x: *const f32, // raw pointer to the contiguous array of vectors
        );

        unsafe fn search_index(
            idx: &mut UniquePtr<IndexACORNFlat>,
            n: i64,                     // number of query vectors
            x: *const f32,              // pointer to an array of the query vectors
            k: i64,                     // number of vectors to return for each query vector
            distances: *mut f32, // pointer to an array of (k*n) floats, each representing a distance of the result from the query vector
            labels: *mut i64, // pointer to an array of (k*n) indices, each representing the ID of the query vector in idx
            filter_id_map: *mut c_char, // a bitmap of the IDs in the filter, an array of (n * N) bools, where N is the total number of vectors in the index, and a '1' represents that the vector at that index passes the predicate for that query.
        );
    }
}

pub mod architecture;
pub mod dataset;
pub mod query;
pub mod stubs;
