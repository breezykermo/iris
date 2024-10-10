mod architecture;
mod dataset;
mod query;
mod stubs;

use crate::architecture::HardwareArchitecture;
use crate::dataset::{Dataset, Deep1B};
use crate::query::{Load, SyncQueries};

use tracing::info;
use tracing_subscriber;

use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("The dataset you seek to benchmark must be trained on a target hardware architecture")]
    UntrainedDataset,
}

// Benchmark function, generic over different datasets and architectures.
pub fn benchmark<D: Dataset, L: Load>(dataset: &D, load: &L) -> Result<(), BenchmarkError> {
    let hw_arch = dataset.get_hardware_architecture();
    if hw_arch.is_none() {
        return Err(BenchmarkError::UntrainedDataset);
    }

    info!("Benchmarking:");
    info!("Dataset: {}", dataset.dataset_info());
    info!("Architecture: {:?}", hw_arch.unwrap());
    info!("Load: {}", load.load_info());

    todo!("Implement the benchmarking logic here");
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let mut dataset = Deep1B::new();

    let load = SyncQueries {
        num_queries: 10_000,
    }; // 10k sync queries

    dataset.train(HardwareArchitecture::SsdStandalone);

    // Run the benchmark
    benchmark(&dataset, &load);
}
