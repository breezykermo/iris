use oak::query::{Load, SyncQueries};

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
pub fn benchmark<L: Load>(load: &L) -> Result<(), BenchmarkError> {
    info!("Benchmarking:");
    info!("Load: {}", load.load_info());

    unimplemented!();
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let args = Args::parse();

    let load = SyncQueries {
        num_queries: 10_000,
    }; // 10k sync queries

    // Run the benchmark
    let benchmark_result = benchmark(&load);
    if let Err(e) = benchmark_result {
        info!("Error: {}", e);
    }

    Ok(())
}
