use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use slog_scope::{debug, info};

use std::time::Instant;
use thiserror::Error;

use oak::bitmask::Bitmask;
use oak::dataset::{OakIndexOptions, SimilaritySearchable};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::router::Router;
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
        level: ConfigLoggingLevel::Info,
    }
    .to_logger("oak-logger")
    .map_err(|e| ExampleError::ServerStartError(e.to_string()))?;

    let _guard = slog_scope::set_global_logger(log.clone());

    let args = Args::parse();

    let mut dataset = FvecsDataset::new(args.dataset, true)?;
    info!("Dataset loaded from disk.");

    let opts = OakIndexOptions {
        gamma: 1,
        m: 32,
        m_beta: 64,
    };

    let _ = dataset.initialize(&opts);
    info!("Seed index constructed.");

    let queries: Vec<_> = (1..7).map(|i| PredicateQuery::new(i)).collect();

    let subdatasets: Vec<_> = queries
        .iter()
        .enumerate()
        .map(|(idx, q)| {
            let idx = idx + 1;
            let mut subdataset = dataset.view(&q);
            let _ = subdataset.initialize(&opts);
            info!("Subindex {idx} as view constructed.");
            subdataset
        })
        .collect();

    let dimensionality = dataset.get_dimensionality() as usize;
    assert_eq!(dimensionality, subdatasets[0].get_dimensionality() as usize);

    // Experiments
    // --------=--
    //
    info!("Constructing random vector to query with {dimensionality} dimensions");
    let query_vector = FlattenedVecs {
        dimensionality,
        data: generate_random_vector(dimensionality),
    };
    let topk = 10;
    let num_queries = query_vector.len();

    let masks: Vec<_> = subdatasets
        .iter()
        .enumerate()
        .map(|(idx, subdataset)| {
            let mask_main = Bitmask::new(&queries[idx], &dataset);
            debug!(
                "Mask main filled: {} / {}",
                mask_main.bitcount(),
                mask_main.capacity()
            );

            let mask_sub = Bitmask::new_full(subdataset);
            debug!(
                "Mask sub filled: {} / {}",
                mask_sub.bitcount(),
                mask_sub.capacity()
            );

            (mask_main, mask_sub)
        })
        .collect();

    let submask_dataset_pairs: Vec<_> = std::iter::zip(&masks, &subdatasets)
        .map(|((mask, _), ds)| (mask, ds))
        .collect();

    let router = Router::new(&dataset, submask_dataset_pairs);

    // Run experiments for Dataset, DatasetPartition, and Router
    for (idx, ((mask_main, mask_sub), subdataset)) in
        std::iter::zip(&masks, &subdatasets).enumerate()
    {
        let idx = idx + 1;
        info!("\n\n---------------------\nExperiment {idx} started:\n---------------------");
        debug!("Searching full dataset for {topk} similar vectors for {num_queries} random query , where attr is equal to 5...");

        let big_start = Instant::now();
        let big_result = dataset.search_with_bitmask(&query_vector, &mask_main, topk, 16);
        let big_end = big_start.elapsed();

        debug!("Searching dataset partition for {topk} similar vectors for {num_queries} random query, with no predicate as we know all vectors match...");

        let small_start = Instant::now();
        let small_result = subdataset.search_with_bitmask(&query_vector, &mask_sub, topk, 16);
        let small_end = small_start.elapsed();

        let big_mean_distance = big_result.unwrap()[0]
            .iter()
            .fold(0, |acc, (distance, _)| acc + distance)
            / topk;

        info!("Results from full search:");
        info!("Mean distance: {:?}", big_mean_distance);
        info!("Time taken: {:?}", big_end);

        let small_mean_distance = small_result.unwrap()[0]
            .iter()
            .fold(0, |acc, (distance, _)| acc + distance)
            / topk;
        info!("Results from sub search:");
        info!("Mean distance: {:?}", small_mean_distance);
        info!("Time taken: {:?}", small_end);

        let routed_start = Instant::now();
        let routed_result = router.search_with_bitmask(&query_vector, &mask_main, topk, 16);
        let routed_end = routed_start.elapsed();

        let routed_mean_distance = routed_result.unwrap()[0]
            .iter()
            .fold(0, |acc, (distance, _)| acc + distance)
            / topk;
        info!("Results from routed search:");
        info!("Mean distance: {:?}", routed_mean_distance);
        info!("Time taken: {:?}", routed_end);
    }

    Ok(())
}
