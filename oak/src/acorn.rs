use crate::dataset::{
    ConstructionError, Dataset, OakIndexOptions, SearchableError, TopKSearchResult,
};
use crate::ffi;
use crate::fvecs::{FlattenedVecs, FvecsDataset, FvecsView};

use std::ffi::c_char;
use tracing::info;

pub struct AcornHnswIndex {
    index: cxx::UniquePtr<ffi::IndexACORNFlat>,
    count: usize,
}

#[cfg(feature = "hnsw_faiss")]
impl AcornHnswIndex {
    pub fn new(
        dataset: &FvecsDataset,
        options: &OakIndexOptions,
    ) -> Result<Self, ConstructionError> {
        let dimensionality = i32::try_from(dataset.get_dimensionality())
            .expect("dimensionality should not be greater than 2,147,483,647");

        let dataset_view = FvecsView::new(&dataset.mmap);
        let fvecs = FlattenedVecs::from(dataset_view);
        let num_fvecs = (fvecs.data.len() as i32) / dimensionality;

        // TODO: is there a way to 'catch' segmentation faults in these unsafe functions?
        let mut index = ffi::new_index_acorn(
            dimensionality,
            options.m,
            options.gamma,
            options.m_beta,
            &dataset.metadata,
        );

        // SAFETY: this is unsafe because we pass a raw ptr to the fvecs data; but we are SURE that
        // we have constructed it appropriately.
        unsafe {
            ffi::add_to_index(&mut index, num_fvecs as i64, fvecs.data.as_ptr());
        }

        Ok(Self {
            index,
            count: num_fvecs as usize,
        })
    }

    pub fn search(
        &mut self,
        query_vectors: FlattenedVecs,
        k: usize,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        let number_of_query_vectors: usize = 1; // TODO: fix this to infer length from FlattenedVecs method
        let length_of_results = k * number_of_query_vectors;
        // TODO: at present there is essentially no filtering, we are just checking that the format
        // is right. A meaningful use of this will require a change to the `search` function
        // signature.
        // let mut filter_id_map: Vec<c_char> = vec![1; number_of_query_vectors * self.count];

        // These two arrays are where the outputs from the cpp methods will be stored
        let mut distances: Vec<f32> = Vec::with_capacity(length_of_results);
        let mut labels: Vec<i64> = Vec::with_capacity(length_of_results);

        let mut filter_id_map: Vec<c_char> =
            Vec::with_capacity(number_of_query_vectors * self.count); // TODO:
                                                                      // create_filter_id_map(metadata, aq, number_of_query_vectors, length_of_results);

        unsafe {
            ffi::search_index(
                &mut self.index,
                number_of_query_vectors as i64,
                query_vectors.data.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                filter_id_map.as_mut_ptr(),
            )
        }

        info!("Search complete");

        unimplemented!();
    }
}
