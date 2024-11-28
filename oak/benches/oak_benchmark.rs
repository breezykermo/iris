use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oak::dataset::{Dataset, OakIndexOptions};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;
use std::sync::Once;

static INIT: Once = Once::new();
static mut DATASET: Option<FvecsDataset> = None;

fn get_dataset() -> &'static FvecsDataset {
    // Only create the index once when benchmarking.
    INIT.call_once(|| {
        let mut dataset = FvecsDataset::new("data/sift_query".to_string()).unwrap();
        let options = OakIndexOptions::default();
        let _ = dataset.initialize(&options);
        unsafe {
            DATASET = Some(dataset);
        }
    });
    let ds = unsafe { DATASET.as_ref().unwrap() };
    ds
}

fn run_query(ds: &FvecsDataset, query: &FlattenedVecs, predicate: &Option<PredicateQuery>) {
    ds.search(query, predicate, 1);
}

fn single_query_no_predicate(c: &mut Criterion) {
    let dataset = get_dataset();
    let predicate = None;

    let dims = dataset.get_dimensionality();
    let query = FlattenedVecs {
        dimensionality: dims,
        data: generate_random_vector(dims),
    };

    c.bench_function("single_query_no_predicate", |b| {
        b.iter(|| run_query(&dataset, &query, &predicate))
    });
}

fn single_query_equals_predicate(c: &mut Criterion) {
    let dataset = get_dataset();
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
    targets = single_query_no_predicate, single_query_equals_predicate
);
criterion_main!(benches);
