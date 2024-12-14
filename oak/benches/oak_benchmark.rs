use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};
use oak::dataset::{OakIndexOptions, SimilaritySearchable, TopKSearchResultBatch};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;
use std::sync::Once;

use std::fs::File;
use std::io::Write;

static INIT: Once = Once::new();
static mut DATASET_GAMMA_1: Option<FvecsDataset> = None;
static mut DATASET_GAMMA_Y: Option<FvecsDataset> = None;

fn load_dataset_gamma_1() -> &'static FvecsDataset {
    // Only create the index once when benchmarking.
    INIT.call_once(|| {
        let mut dataset = FvecsDataset::new("data/sift_base".to_string()).unwrap();
        let options = OakIndexOptions::default();
        let _ = dataset.initialize(&options);
        unsafe {
            DATASET_GAMMA_1 = Some(dataset);
        }
    });
    let ds = unsafe { DATASET_GAMMA_1.as_ref().unwrap() };
    ds
}

fn load_dataset_gamma_y() -> &'static FvecsDataset {
    // Only create the index once when benchmarking.
    INIT.call_once(|| {
        let mut dataset = FvecsDataset::new("data/sift_base".to_string()).unwrap();
        let options = OakIndexOptions {
            gamma: 12,
            m: 32,
            m_beta: 64,
        };
        let _ = dataset.initialize(&options);
        unsafe {
            DATASET_GAMMA_1 = Some(dataset);
        }
    });
    let ds = unsafe { DATASET_GAMMA_1.as_ref().unwrap() };
    ds
}

fn run_query(
    ds: &FvecsDataset,
    query: &FlattenedVecs,
    predicate: &Option<PredicateQuery>,
) -> TopKSearchResultBatch {
    ds.search(query, predicate, 1).unwrap()
}

fn single_query_no_predicate_gamma_1(c: &mut Criterion) {
    let dataset = load_dataset_gamma_1();

    let queries = FvecsDataset::new("data/sift_query".to_string()).unwrap();
    // let predicates = Vec::<PredicateQuery>::from(&queries);
    // NOTE: passing all 10000 queries at the same time with a non-None query throughs a segfault,
    // presumably as the bitmask generation is too memory intensive.
    let queries = FlattenedVecs::from(&queries);

    // let first_predicate = predicates.get(0).cloned();
    // let first_predicate: Option<PredicateQuery> = Some(PredicateQuery::new(10));
    let first_predicate: Option<PredicateQuery> = None;
    // debug!("Predicate is: {first_predicate}");

    c.bench_function("single_query_no_predicate_gamma_1", |b| {
        b.iter(|| run_query(&dataset, &queries, &first_predicate))
    });
}

// NOTE: This benchmark is likely slower due to the way that the bitmask is generated. To properly
// benchmark the index, we would need to call `search` at a lower level.
fn single_query_equals_predicate_gamma_1(c: &mut Criterion) {
    let dataset = load_dataset_gamma_1();
    let predicate = Some(PredicateQuery::new(1));

    let dims = dataset.get_dimensionality();
    let query = FlattenedVecs {
        dimensionality: dims,
        data: generate_random_vector(dims),
    };

    c.bench_function("single_query_equals_predicate", |b| {
        b.iter(|| run_query(&dataset, &query, &predicate))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = single_query_no_predicate_gamma_1, single_query_equals_predicate_gamma_1
);
criterion_main!(benches);
