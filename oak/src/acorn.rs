use crate::dataset::{
    ConstructionError, Dataset, OakIndexOptions, SearchableError, TopKSearchResult,
};
use crate::ffi;
use crate::fvecs::{FlattenedVecs, FvecsDataset};
use crate::predicate::PredicateQuery;

use core::ffi::c_char;
// use tracing::{debug, info};
use slog_scope::{debug, info};

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

        let mut index = ffi::new_index_acorn(
            dimensionality,
            options.m,
            options.gamma,
            options.m_beta,
            &dataset.metadata,
        );
        debug!(
            "Constructed index with dimensionality: {dimensionality}, m: {}, gamma: {}, m_beta: {}",
            options.m, options.gamma, options.m_beta
        );

        let fvecs = FlattenedVecs::from(dataset);
        let num_fvecs = dataset.len();
        debug!("Adding {num_fvecs} vectors to the index...");

        // SAFETY: this is unsafe because we pass a raw ptr to the fvecs data; but we are SURE that
        // we have constructed it appropriately.
        // TODO: is there a way to 'catch' segmentation faults in these unsafe functions?
        unsafe {
            ffi::add_to_index(&mut index, num_fvecs as i64, fvecs.data.as_ptr());
        }
        debug!("Added {num_fvecs} vectors to the index.");

        Ok(Self {
            index,
            count: num_fvecs as usize,
        })
    }

    pub fn search(
        &mut self,
        query_vectors: &FlattenedVecs,
        filter_id_map: &mut Vec<c_char>,
        k: usize,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        let number_of_query_vectors: usize = query_vectors.len();
        debug!("Searching queries: {number_of_query_vectors} in batch.");
        let length_of_results = k * number_of_query_vectors;
        debug!("Length of results arrays: {length_of_results}.");

        // These two arrays are where the outputs from the cpp methods will be stored
        let mut distances: Vec<f32> = Vec::with_capacity(length_of_results);
        let mut labels: Vec<i64> = Vec::with_capacity(length_of_results);

        let filter_id_map_length = filter_id_map.len();
        debug!("Length of bitmap representing predicate: {filter_id_map_length}.");

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
