use oak::dataset::{
    AcornHnswIndex, AcornHnswOptions, Dataset, FvecsDataset, Searchable, VectorIndex,
};

use clap::Parser;

use tracing::info;
use tracing_subscriber;

use anyhow::Result;
use thiserror::Error;

// Ensure that only one of FAISS or hnsw_rs is used.
#[cfg(all(feature = "hnsw_faiss", feature = "hnsw_rust"))]
compile_error!(
    "Features `hnsw_faiss` and `hnsw_rust` cannot be enabled at the same time. Please enable only one."
);

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Generic error")]
    GenericError,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, required(true))]
    dataset: String,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let args = Args::parse();

    let dataset = FvecsDataset::new(args.dataset)?;
    info!("Dataset loaded from disk.");

    // NOTE: These parameters are taken from https://github.com/csirianni/ACORN/blob/main/README.md
    let opts = AcornHnswOptions {
        gamma: 12,
        m: 32,
        m_beta: 32,
    };

    let main_index = AcornHnswIndex::new(&dataset, &opts);
    info!("Seed index constructed.");

    info!("Open for connections.");
    loop {
        // TODO:
    }

    Ok(())
}
