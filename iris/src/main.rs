mod architecture;
mod dataset;
mod query;
mod stubs;

use crate::architecture::HardwareArchitecture;
use crate::dataset::{Dataset, Deep1X, Searchable, VectorIndex};
use crate::query::{Load, SyncQueries};

use clap::Parser;

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
    info!("Benchmarking:");
    info!("Dataset: {}", dataset.dataset_info());
    info!("Architecture: {:?}", dataset.get_hardware_architecture());
    info!("Load: {}", load.load_info());

    unimplemented!();
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, required(true))]
    architecture: HardwareArchitecture,

    #[arg(short, long, required(true))]
    cluster_size: usize,

    #[arg(short, long, required(true))]
    node_num: usize,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let args = Args::parse();

    let mut dataset = Deep1X::new(args.architecture, args.cluster_size, args.node_num)?;
    let _ = dataset.build_index(VectorIndex::IndexFlatL2)?;

    // let mut dataset = StubVectorDataset::new();

    let load = SyncQueries {
        num_queries: 10_000,
    }; // 10k sync queries

    // Run the benchmark
    let benchmark_result = benchmark(&dataset, &load);
    if let Err(e) = benchmark_result {
        info!("Error: {}", e);
    }

    Ok(())
}
