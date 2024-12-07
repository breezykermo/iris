use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use futures_util::{
    future::{select, Either},
    stream::{FuturesUnordered, Stream, StreamExt, TryStreamExt},
};
use slog_scope::{debug, info};
use core::time;
use std::time::Duration;
use thiserror::Error;

use oak::dataset::{Dataset, OakIndexOptions, SearchableError, TopKSearchResultBatch};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::poisson::SpinTicker;
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

struct ExpResult {
    durs: Vec<Duration>,
    remaining_inflight: usize,
    tot_time: Duration,
    num_clients: usize,
}

struct Op {
    dataset: &FvecsDataset,
    query_vector: &FlattenedVecs,
    filter_id_map: &Vec<c_char>,
    k: usize,
}
impl Op {
    fn exec(&self) -> Result<TopKSearchResultBatch, SearchableError> {
        self.dataset
            .index
            .as_ref()
            .unwrap()
            .search(self.query_vector, self.filter_id_map, self.k)
    }
}

struct QueryStats {
    query_op: Op,
    recall: f64,
    latency: Duration
}
/// time_req measures time taken for the vectorDB call
fn time_req(op: Op) -> Result<(Duration, Result<TopKSearchResultBatch, SearchableError>), Report> {
    let now = tokio::time::Instant::now();
    let result = op.exec();
    Ok((now.elapsed(), result))
}
///Alt: compute gt for each query and log it into a CSV and then separately 
/// run these to compare against pre-made gt
/// Unsure if this is actually correct but keeping it around for reference
// fn calculate_recall(search_result: &TopKSearchResultBatch, ground_truth: &GroundTruth) -> (f64, f64, f64) {
//     let gt = calculate_gt(ground_truth);
//     let nq = gt.len();
//     let mut n_1 = 0;
//     let mut n_10=0;
//     let mut n_100 = 0;

//     for i in 0..nq {
//         let gt_nn = gt[i*k]; // top 1 search
//         for j in 0..k {
//             if j <1 {
//                 n_1 += 1;
//             }
//             if j < 10 {
//                 n_10 +=1;
//             }
//             if j < 100 {
//                 n_100 += 1;
//             }
//         }
//     }
//     (n_1 as f64 / nq as f64, 
//         n_10 as f64 / nq as f64, 
//         n_100 as f64 / nq as f64)
// }


async fn calculate_recall(gt: Vec<FlattenedVecs>, acorn_index: Vec<FlattenedVecs>, k: usize) {
    // Some way of getting groundtruth -TODO: check FAISS for it
    // Then decide the type of the input. Some Vec of Vec
    nq = acorn_index.len();
    gt = calculate_gt(gt);
    let mut n_1 = 0;
    let mut n_10= 0;
    let mut n_100 = 0;
    let mut rank = 1;
    let mut results = vec![];
    while rank <= k {
        let gt_nn = &gt[0..rank]; // top-rank queries
        let retrieved = &acorn_index[0..rank];
        
        // Compute intersection
        let intersection = retrieved.iter().filter(|&&id| gt_nn.contains(&id)).count();
        let recall = intersection as f64;
        
        recall /= rank as f64;
        results.push(recall);
        rank *= 10;
    }
    results // Recall for top-1, top-10 and top-100
}

/// This function is needed because the SIFT dataset doesn't have predicates. The
/// predicates have been generated and uniformly distributed across a space. Now
/// when we consult groundtruth indices, the top-k queries may not match the 
/// required predicate and thus, shouldn't be counted in recall. Instead, we 
/// take the top-k queries that match the predicate and compare against 
/// OAK-generated results.
fn calculate_gt(gt_index: Vec<FlattenedVecs>)-> Vec<FlattenedVecs> {
    // TODO: Need Lachlan's help on loading in predicates and searching by them
    // Is there something modular? Where are pedicates generated  (in case we
    // distribute in range 1-30 or add more predicates to search by or sth)
    vec![]
}

/// Loading the groundtruth is simply loading in the fvecs file. The complexity
/// of checking predicate matches is handed over to the calculate recall 
fn load_gt(path: &str) ->  {
    // TODO: Parse CSV into a usable GroundTruth data structure
    Ok(GroundTruth {})
}

/// This query loop essentially takes a query load, finds the latency of running
/// the queries one at a time, looks at the total time taken and determines QPS
/// and finally, logs the recall for each query as compared to the groundtruth
/// indices
fn query_loop (
    ops: Vec<Op> // each Op knows 
) -> Result<(), Report> {
    let benchmark_results = vec![];
    for op in ops {
        match time_req(op) {
            Ok((latency, Ok((res, err)))) => {
                let recall = calculate_recall(gt, res);
                benchmark_results.push(QueryStats(
                    op,
                    recall,
                    latency
                ))
            }
            Ok((_, Err(e))) => {println!("Search operation failed with error {:?}", e)}
            Err(e) => {println!("Timing operation failed with error {:?}", e)}
        }
    }
    Ok(())
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

    let _ = dataset.initialize(&opts);
    info!("Seed index constructed.");
    // Load queries
    // Load predicates
    // Load GT
    // pop from both, run queryloop 
    // log latency, operation (query vec and predicate searched for) and recall

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
