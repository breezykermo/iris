use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use oak::dataset::TopKSearchResult;
use oak::predicate::PredicateOp;
use oak::bitmask::Bitmask;
use std::time::Duration;
use thiserror::Error;
use slog_scope::{debug, info};
use oak::dataset::{Dataset, OakIndexOptions, SearchableError, TopKSearchResultBatch};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use csv::Writer;
use csv::Reader;
use core::ffi::c_char;
use std::env;

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
struct ExpResult {
    durs: Vec<Duration>,
    remaining_inflight: usize,
    tot_time: Duration,
    num_clients: usize,
}

struct QueryStats {
    recall_1: bool,
    recall_10: bool,
    recall_100: bool,
    latency: Duration
}

fn averages(queries: Vec<QueryStats>) -> Result<(f32, f32, f32, f32)> {
    let total_latencies:f32 = queries.iter().map(|qs| qs.latency).sum::<Duration>().as_secs() as f32;
    let total_r1:f32 = queries.iter().map(|qs| qs.recall_1).count() as f32;
    let total_r10:f32 = queries.iter().map(|qs| qs.recall_10).count() as f32;
    let total_r100:f32 = queries.iter().map(|qs| qs.recall_100).count() as f32;
    let count:f32 = queries.len() as f32;
    Ok((total_latencies as f32 /count as f32, total_r1 as f32 /count as f32, total_r10 as f32 /count as f32, total_r100 as f32 /count as f32))
}

fn calculate_recall_1(gt: &Vec<usize>, acorn_result: TopKSearchResultBatch) -> Result<(f32)> {
    // Figure out how to represent the Groundtruth and index into it!!
    let mut n_1: usize= false;
    let mut n_10: usize = false;
    for (query_index, neighbors) in acorn_result.iter().enumerate() {
        let groundNN = gt[query_index];
        for (i, j) in neighbors.iter().enumerate() {
            if j.0 == groundNN {
                if i <1 {
                    n_1 += 1;
                }
                if i < 10 {
                    n_10 += 1;
                }
                break;
            }
        }
    }
    Ok(n_10 as f32/ gt.len() as f32)
}

fn read_csv(file_path: &str) -> Result<Vec<usize>> {
    let mut rdr = Reader::from_path(file_path)?;
    let mut values = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let value: usize = record[0].parse::<usize>()?; // why is this needed?
        values.push(value);
    }
    Ok(values)
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
    // How are they learning about dataset predicates?

    let _ = dataset.build_index(&opts);
    info!("Seed index constructed.");

    let query = PredicateQuery::new(1);

    let mut subdataset = dataset.view(&query);
    let _ = subdataset.build_index(&opts);
    info!("Subindex as view constructed.");

    let dimensionality = dataset.get_dimensionality() as usize;
    assert_eq!(dimensionality, subdataset.get_dimensionality() as usize);

    let mut dataset = FvecsDataset::new(args.query)?;
    info!("Query set loaded from disk.");

    let topk = 10;
    let num_queries = query_vector.len();

    let mask_main = Bitmask::new(&query, &dataset);
    let mask_sub = Bitmask::new_full(&subdataset);

    info!("Searching full dataset for {topk} similar vectors for {num_queries} random query , where attr is equal to 5...");

    let now = tokio::time::Instant::now();
    let result = dataset.search_with_bitmask(&query_vector, mask_main, topk);
    let end = now.elapsed();

    let latency = end / num_queries;
    info!("QPS is {latency}");

    info!("GT loading...");
    let groundtruth_path = "data/outdir/sift_groundtruth.csv";
    // let variable_gt_path = "./outdir/sift_groundtruth.csv";
    let gt = read_csv(groundtruth_path);

    let recall = calculate_recall_1(gt, result);

    info!("Recall@10 is {recall}");


    // Log results to CSV
    // let mut wtr = Writer::from_path("output.csv")?;
    // writer.write_record(&["Average Latency (s)", "Recall@K"])?;
    // for d in latencies {
    //     writer.write_record(&[d.as_secs().to_string()])?;
    // }
    // writer.flush()?;

    // let dimensionality = dataset.get_dimensionality() as usize;
    // info!("Constructing random vector to query with {dimensionality} dimensions");
    // let query_vector = FlattenedVecs {
    //     dimensionality,
    //     data: generate_random_vector(dimensionality),
    // };
    // let topk = 10;
    // let num_queries = query_vector.len();
    // let query = Some(PredicateQuery::new(5));
    //
    // info!("Searching for {topk} similar vectors for {num_queries} random query, where attr is equal to 5...");
    //
    // let result = dataset.search(&query_vector, &query, topk);
    //
    // info!("Got results.");
    // info!("{:?}", result);

    Ok(())
}

// Run.py 1 number for each vector and then 