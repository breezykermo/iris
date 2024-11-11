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
    dataset_fname: String,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let args = Args::parse();

    let mut dataset = FvecsDataset::new(args.dataset_fname)?;

    let opts = AcornHnswOptions {
        m: 1,
        gamma: 1,
        m_beta: 1,
    };

    let main_index = AcornHnswIndex::new(&dataset, &opts);

    loop {
        todo!("Process requests")
    }

    Ok(())
}
