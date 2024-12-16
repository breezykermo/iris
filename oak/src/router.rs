use crate::bitmask::Bitmask;
use crate::dataset::SimilaritySearchable;
use crate::fvecs::FvecsDatasetPartition;
use slog_scope::debug;

pub struct Router<'a> {
    base: &'a dyn SimilaritySearchable,
    opportunistic: Vec<(&'a Bitmask, &'a FvecsDatasetPartition<'a>)>,
    search_masks: Vec<Bitmask>,
}

impl<'a> Router<'a> {
    pub fn new(
        base: &'a dyn SimilaritySearchable,
        opportunistic: Vec<(&'a Bitmask, &'a FvecsDatasetPartition<'a>)>,
    ) -> Self {
        let search_masks: Vec<Bitmask> = opportunistic
            .iter()
            .map(|(_, partition)| Bitmask::new_full(*partition))
            .collect();

        Router {
            base,
            opportunistic,
            search_masks,
        }
    }
}

impl SimilaritySearchable for Router<'_> {
    fn len(&self) -> usize {
        self.base.len()
    }

    fn get_dimensionality(&self) -> usize {
        self.base.get_dimensionality()
    }

    fn get_metadata(&self) -> &crate::dataset::HybridSearchMetadata {
        self.base.get_metadata()
    }

    fn initialize(
        &mut self,
        _opts: &crate::dataset::OakIndexOptions,
    ) -> anyhow::Result<(), crate::dataset::ConstructionError> {
        // TODO: do we need to do anything in here to optimize bitmask comparisons?
        Ok(())
    }

    fn search(
        &self,
        query_vectors: &crate::fvecs::FlattenedVecs,
        predicate_query: &Option<crate::predicate::PredicateQuery>,
        topk: usize,
        efsearch: i64,
    ) -> anyhow::Result<Vec<crate::dataset::TopKSearchResult>, crate::dataset::SearchableError>
    {
        self.base
            .search(query_vectors, predicate_query, topk, efsearch)
    }

    fn search_with_bitmask(
        &self,
        query_vectors: &crate::fvecs::FlattenedVecs,
        query_bitmask: &crate::bitmask::Bitmask,
        topk: usize,
        efsearch: i64,
    ) -> anyhow::Result<Vec<crate::dataset::TopKSearchResult>, crate::dataset::SearchableError>
    {
        let base_meta = self.base.get_metadata();
        let base_meta_len = base_meta.len() as f32;

        let (best_index, best_score) = self
            .opportunistic
            .iter()
            .map(|(opp_mask, opp_index)| {
                let opp_meta = opp_index.get_metadata();
                let perf_gain = base_meta_len / (opp_meta.len() as f32);
                // debug!("Performance gain: {}", perf_gain);
                let recall_loss = query_bitmask.jaccard_similarity(opp_mask);
                // debug!("Recall loss: {}", recall_loss);
                // Recall is a maximum of 1 (if masks perfectly overlap).
                (perf_gain as f64) * recall_loss
            })
            // get the index of the max
            .enumerate()
            .max_by(|&(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // debug!(
        //     "The best opportunistic index is at position {} with a score of {}",
        //     best_index, best_score
        // );

        // TODO: if using the subindex, shouldn't use `query_bitmask` but rather should use
        // mask_sub equivalent

        let score_threshold = 10.;
        if best_score > score_threshold {
            let (_, opp_index) = self.opportunistic[best_index];
            let search_mask = &self.search_masks[best_index];
            let search_results = opp_index
                .search_with_bitmask(query_vectors, search_mask, topk, efsearch)
                .unwrap();
            // if it uses an opportunistic index, we need to map those inner indexes back to
            // indexes over the global search space
            let mapped_results: Vec<(usize, f32)> = search_results
                .get(0)
                .unwrap()
                .iter()
                .map(|(inner_index, distance)| {
                    (
                        opp_index.original_indices.as_ref().unwrap()[inner_index.clone()],
                        distance.clone(),
                    )
                })
                .collect();
            Ok(vec![mapped_results])
        } else {
            self.base
                .search_with_bitmask(query_vectors, query_bitmask, topk, efsearch)
        }
    }
}
