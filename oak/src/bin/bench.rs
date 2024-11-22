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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let args = Args::parse();

    // TODO: run the benchmark

    Ok(())
}
