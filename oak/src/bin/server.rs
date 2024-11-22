use oak::dataset::{Dataset, OakIndexOptions};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::stubs::generate_random_vector;

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

    let mut dataset = FvecsDataset::new(args.dataset)?;
    info!("Dataset loaded from disk.");

    let opts = OakIndexOptions {
        gamma: 12,
        m: 32,
        m_beta: 32,
    };

    let _ = dataset.initialize(&opts);
    info!("Seed index constructed.");

    let dimensionality = dataset.get_dimensionality() as usize;
    let query_vector = FlattenedVecs {
        dimensionality,
        data: generate_random_vector(dimensionality),
    };
    info!("Searching for similar vectors to a random vector...");

    let result = dataset.search(query_vector, 3);

    // let results = dataset.search(xq, 10);
    // info!("Open for connections.");
    // loop {
    //     // TODO:
    // }

    Ok(())
}
