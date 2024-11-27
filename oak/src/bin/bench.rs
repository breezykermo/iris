use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use slog_scope::{debug, info};
use thiserror::Error;

use oak::dataset::{Dataset, OakIndexOptions};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;

// Ensure that only one of FAISS or hnsw_rs is used.
#[cfg(all(feature = "hnsw_faiss", feature = "hnsw_rust"))]
compile_error!(
    "Features `hnsw_faiss` and `hnsw_rust` cannot be enabled at the same time. Please enable only one."
);

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Generic error")]
    GenericError,
    #[error("Failed to start server: {0}")]
    ServerStartError(String),
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, required(true))]
    dataset: String,
}

fn main() -> Result<()> {
    let log = ConfigLogging::StderrTerminal {
        level: ConfigLoggingLevel::Debug,
    }
    .to_logger("oak-logger")
    .map_err(|e| ServerError::ServerStartError(e.to_string()))?;

    let _guard = slog_scope::set_global_logger(log.clone());

    let args = Args::parse();

    let mut dataset = FvecsDataset::new(args.dataset)?;
    info!("Dataset loaded from disk.");

    let opts = OakIndexOptions {
        gamma: 1,
        m: 32,
        m_beta: 64,
    };

    let _ = dataset.initialize(&opts);
    info!("Seed index constructed.");

    let dimensionality = dataset.get_dimensionality() as usize;
    info!("Constructing random vector to query with {dimensionality} dimensions");
    let query_vector = FlattenedVecs {
        dimensionality,
        data: generate_random_vector(dimensionality),
    };
    let topk = 10;
    let num_queries = query_vector.len();
    info!("Searching {topk} similar vectors for {num_queries} queries...");

    let query: Option<PredicateQuery> = None;

    // let result = dataset.search(query_vector, query, topk);

    // let results = dataset.search(xq, 10);
    info!("Open for connections.");
    loop {
        // TODO:
    }

    Ok(())
}
