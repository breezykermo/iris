use crate::dataset::{
    ConstructionError, HybridSearchMetadata, OakIndexOptions, SearchableError, TopKSearchResult,
};
use crate::ffi;
use crate::fvecs::FlattenedVecs;

use core::ffi::c_char;
use slog_scope::debug;

#[allow(dead_code)]
pub struct AcornHnswIndex {
    index: cxx::UniquePtr<ffi::IndexACORNFlat>,
    count: usize,
}

#[cfg(feature = "hnsw_faiss")]
impl AcornHnswIndex {
    pub fn new(
        dimensionality: i32,
        metadata: &HybridSearchMetadata,
        flattened: &FlattenedVecs,
        options: &OakIndexOptions,
    ) -> Result<Self, ConstructionError> {
        let mut index = ffi::new_index_acorn(
            dimensionality,
            options.m,
            options.gamma,
            options.m_beta,
            metadata.as_ref(),
        );
        // debug!(
        //     "Constructed index with dimensionality: {dimensionality}, m: {}, gamma: {}, m_beta: {}",
        //     options.m, options.gamma, options.m_beta
        // );

        // NOTE: this brings the data into memory.
        let num_fvecs = flattened.len();

        // assert_eq!(flattened.len(), metadata.len(), "When constructing a new index, there must be one and only one piece of metadata for each vector");

        // debug!("Adding {num_fvecs} vectors to the index...");

        // SAFETY: this is unsafe because we pass a raw ptr to the fvecs data; but we are SURE that
        // we have constructed it appropriately.
        // TODO: is there a way to 'catch' segmentation faults in these unsafe functions?
        unsafe {
            ffi::add_to_index(&mut index, num_fvecs as i64, flattened.data.as_ptr());
        }
        // debug!("Added {num_fvecs} vectors to the index.");

        Ok(Self {
            index,
            count: num_fvecs as usize,
        })
    }

    pub fn search(
        &self,
        query_vectors: &FlattenedVecs,
        filter_id_map: &mut Vec<c_char>,
        k: usize,
        efsearch: i64,
    ) -> Result<Vec<TopKSearchResult>, SearchableError> {
        let number_of_query_vectors: usize = query_vectors.len();
        // debug!("Searching queries: {number_of_query_vectors} in batch.");
        let length_of_results = k * number_of_query_vectors;
        // debug!("Length of results arrays: {length_of_results}.");

        // These two arrays are where the outputs from the cpp methods will be stored
        let mut distances: Vec<f32> = vec![0 as f32; length_of_results];
        let mut labels: Vec<i64> = vec![0; length_of_results];

        let filter_id_map_length = filter_id_map.len();
        // debug!("Length of bitmap representing predicate: {filter_id_map_length}.");

        unsafe {
            ffi::search_index(
                &self.index,
                number_of_query_vectors as i64,
                query_vectors.data.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                filter_id_map.as_mut_ptr(),
                efsearch,
            )?
        }

        // debug!("Search complete");

        let combined: Vec<(usize, f32)> = labels
            .into_iter()
            .map(|i| i as usize)
            .zip(distances)
            .collect();
        Ok(combined.chunks(k).map(|chunk| chunk.to_vec()).collect())
    }
}
