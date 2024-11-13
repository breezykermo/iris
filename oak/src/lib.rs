#[cxx::bridge(namespace = "faiss")]
pub mod ffi {
    unsafe extern "C++" {
        include!("oak/third_party/ACORN/faiss/IndexACORN.h");

        type IndexACORNFlat;

        fn new_index_acorn(d: i32, M: i32, gamma: i32, M_beta: i32) -> UniquePtr<IndexACORNFlat>;

        // unsafe fn add(
        //     &self,
        //     n: i64,        // number of vectors to be added
        //     x: *const f32, // raw pointer to the contiguous array of vectors
        // );
    }
}

pub mod architecture;
pub mod dataset;
pub mod query;
pub mod stubs;
