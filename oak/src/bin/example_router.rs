use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use slog_scope::info;
use thiserror::Error;

use oak::dataset::{Dataset, OakIndexOptions};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;

#[derive(Error, Debug)]
pub enum ExampleError {
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
    .map_err(|e| ExampleError::ServerStartError(e.to_string()))?;

    let _guard = slog_scope::set_global_logger(log.clone());

    let args = Args::parse();

    let mut dataset = FvecsDataset::new(args.dataset)?;
    info!("Dataset loaded from disk.");

    let opts = OakIndexOptions {
        gamma: 1,
        m: 32,
        m_beta: 64,
    };

    let _ = dataset.build_index(&opts);
    info!("Seed index constructed.");

    let query = PredicateQuery::new(5);

    let mut subindex = dataset.view(&query);
    let _ = subindex.build_index(&opts);
    info!("Subindex as view constructed.");

    let dimensionality = dataset.get_dimensionality() as usize;
    assert_eq!(dimensionality, subindex.get_dimensionality() as usize);

    info!("Constructing random vector to query with {dimensionality} dimensions");
    let query_vector = FlattenedVecs {
        dimensionality,
        data: generate_random_vector(dimensionality),
    };
    let topk = 10;
    let num_queries = query_vector.len();

    info!("Searching for {topk} similar vectors for {num_queries} random query, where attr is equal to 5...");

    let result = dataset.search(&query_vector, &Some(query), topk);

    info!("Got results.");
    info!("{:?}", result);

    Ok(())
}
