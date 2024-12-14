use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use oak::bitmask::Bitmask;
use std::time::Duration;
use thiserror::Error;
use slog_scope::info;
use oak::dataset::{Dataset, OakIndexOptions, SearchableError, TopKSearchResultBatch};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use csv::ReaderBuilder;
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
    #[arg(short, long, required(true))]
    query: String,
    #[arg(short, long, required(true))]
    groundtruth: String,
}
struct ExpResult {
    durs: Vec<Duration>,
    remaining_inflight: usize,
    tot_time: Duration,
    num_clients: usize,
}

struct QueryStats {
    recall_10: bool,
    latency: u128 //currently representing milliseconds, need to rework since its per query
}

fn query_loop(
    dataset: FvecsDataset,
    queries : Vec<FlattenedVecs>,
    bitmask : &Bitmask,
    k: usize,
    gt: Vec<usize>
) -> Result<Vec<QueryStats>> {
    let mut results = vec![];
    info!{"We have {} queries and {} gt elts", queries.len(), gt.len()};
    for (i, q) in queries.iter().enumerate() {
        let now = tokio::time::Instant::now();
        let result = dataset.search_with_bitmask(&q, &bitmask, k)?;
        let end = now.elapsed();
        let latency = end.as_millis();
        let r10 = calculate_recall_1(gt[i], result)?;
        results.push(QueryStats{
            recall_10: r10,
            latency: latency
        });

    }
    Ok(results)
}
fn averages(queries: Vec<QueryStats>) -> Result<(f64, f64, f64)> {
    let total_latencies:f64 = queries.iter().map(|qs| qs.latency).sum::<u128>() as f64;
    let total_r10:f64 = queries.iter().filter(|qs| qs.recall_10).count() as f64;
    info!("TOTAL R10: {}", total_r10);
    let count:f64 = queries.len() as f64;
    Ok((
        total_latencies,
        total_latencies /count, 
        total_r10 /count, 
    ))
}

fn calculate_recall_1(gt: usize, acorn_result: TopKSearchResultBatch) -> Result<bool> {
    let mut n_10: bool = false;
    for (i, j) in acorn_result[0].iter().enumerate() {
        // topk should be 10 so this loop should be bounded at 10 per query
        if j.0 == gt {
            info!("{} == {} recall", j.0, gt);
            if i < 10 {
                n_10 = true;
            }
            break;
            // we can break out whenever the matching index was found
        }
    }
    if !n_10 {
        info!("HIIIIII");
    }
    Ok(n_10)
}

fn read_csv(file_path: String) -> Result<Vec<usize>> {
    let mut rdr = ReaderBuilder::new()
                    .has_headers(false)
                    .from_path(file_path)?;
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

    let mut dataset = FvecsDataset::new(args.dataset, true)?;
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

    let mut query_set = FvecsDataset::new(args.query, false)?;
    let batched_queries = FlattenedVecs::from(&query_set);
    info!("Query set loaded from disk.");

    let topk = 10;
    let num_queries = batched_queries.len();
    info!("Total {num_queries} queries loaded");
    let queries = batched_queries.to_vec();
    assert_eq!(queries.len(), 100);
    info!("Converted into {}", queries.len());

    let mask_main = Bitmask::new(&query, &dataset);
    let mask_sub = Bitmask::new_full(&subdataset);
    let v = mask_main.map.len();
    let s = mask_sub.map.len();
    info!("{}", v);
    info!("{}", s);

    info!("GT loading...");
    // let variable_gt_path = "./outdir/sift_groundtruth.csv";
    let gt = read_csv(args.groundtruth)?;
    info!("{} gt queries found", gt.len());

    info!("Searching full dataset for {topk} similar vectors for {num_queries} random query , where attr is equal to 1...");
    let qs = query_loop(dataset, queries, &mask_main, topk, gt)?;
    match averages(qs) {
        Ok((lat, qps, r10)) => {
            info!("QPS was {qps} milliseconds with total latency
             being {lat} for {num_queries} and Recall@10 was {r10}")
        }
        Err(_) => {
            info!("Error calculating averages")
        }
    }
    info!("Got results.");
    Ok(())
}