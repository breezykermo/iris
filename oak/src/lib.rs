#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("oak/third_party/ACORN/faiss/IndexACORN.h");

        type IndexACORN;

        fn new_index_acorn(
            d: i32,
            M: i32,
            gamma: i32,
            metadata: &CxxVector<i32>,
            M_beta: i32,
        ) -> UniquePtr<IndexACORN>;

        // unsafe fn add(&self, n: i64, x: *const f32);
    }
}

pub mod architecture;
pub mod dataset;
pub mod query;
pub mod stubs;
