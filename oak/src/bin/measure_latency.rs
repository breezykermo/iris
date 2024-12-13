use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use oak::dataset::TopKSearchResult;
use oak::predicate::PredicateOp;
use slog_scope::{debug, info};
use core::time;
use std::time::Duration;
use thiserror::Error;

use oak::dataset::{Dataset, OakIndexOptions, SearchableError, TopKSearchResultBatch};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;
use csv::Writer;
use core::ffi::c_char;

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

// struct Op {
//     dataset: &FvecsDataset,
//     query_vector: &FlattenedVecs,
//     filter_id_map: &Vec<c_char>,
//     k: usize,
// }
// impl Op {
//     fn exec(&self) -> Result<TopKSearchResultBatch, SearchableError> {
//         self.dataset
//             .index
//             .as_ref()
//             .unwrap()
//             .search(self.query_vector, self.filter_id_map, self.k)
//     }
// }

struct QueryStats {
    query_op: Op,
    recall: f64,
    latency: Duration
}
/// time_req measures time taken for the vectorDB call
fn time_req(
    dataset: &FvecsDataset,
    query_vector: &FlattenedVecs,
    filter_id_map: &Vec<c_char>,
    k: usize
) -> Result<(Duration, Result<TopKSearchResultBatch, SearchableError>), Report> {
    let now = tokio::time::Instant::now();
    let result = dataset.search_with_bitmask(&query_vector, filter_id_map, k);
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


async fn calculate_recall(gt: Vec<FlattenedVecs>, acorn_index: Vec<FlattenedVecs>, k: usize) -> f64 {
    // Some way of getting groundtruth -TODO: check FAISS for it
    // Then decide the type of the input. Some Vec of Vec
    nq = acorn_index.len();
    let mut results = vec![];
    let gt_nn = &gt[0..k]; // top-rank queries
    let retrieved = &acorn_index[0..k];
    
    // Compute intersection
    let intersection = retrieved.iter().filter(|&&id| gt_nn.contains(&id)).count();
    intersection as f64 / k as f64
}



/// This query loop essentially takes a query load, finds the latency of running
/// the queries one at a time, looks at the total time taken and determines QPS
/// and finally, logs the recall for each query as compared to the groundtruth
/// indices
fn query_loop (
    dataset: FvecsDataset,
    query_vectors: FlattenedVecs, // TODOM: Ask Lachlan whether its Vec of or not
    filter_id_map: Vec<c_char>,
    k: usize,
) -> Result<Vec<Duration>> {
    let benchmark_results = vec![];
    for op in query_vectors {
        match time_req(&dataset, &op, &filter_id_map, k) {
            Ok((latency, Ok((res, err)))) => {
                // let recall = calculate_recall(gt, res);
                // benchmark_results.push(QueryStats(
                //     None,
                //     None,
                //     latency
                // ))
                benchmark_results.push(latency);
            }
            Ok((_, Err(e))) => {println!("Search operation failed with error {:?}", e)}
            Err(e) => {println!("Timing operation failed with error {:?}", e)}
        }
    }
    info!("Completed benchmarking queries!");
    Ok(benchmark_results)
}

fn average_duration(latencies: Vec<Duration>) -> Duration {
    let total = latencies.iter().sum();
    let count = durations.len();
    if count == 0 {
        Duration::ZERO
    } else {
        total / count as u32
    }
}

fn calculate_recall_1(gt: usize, acorn_result: TopKSearchResultBatch) -> Result<(f32, f32, f32)> {
    todo!(); // Ask lachlan how to iterate through TopKSearchResbatch
    // Figure out how to represent the Groundtruth and index into it!!
    nq = gt.len();

    let mut n_1 = 0;
    let mut n_10=0;
    let mut n_100 = 0;

    for i in 0..nq {
        let gt_nn = gt[i*k]; // top 1 search
        for j in 0..k {
            if j <1 {
                n_1 += 1;
            }
            if j < 10 {
                n_10 +=1;
            }
            if j < 100 {
                n_100 += 1;
            }
        }
    }
    let r_1 = n_1 as f64/ nq as f64;
    let r_10 = n_10 as f64/ nq as f64;
    let r_100 = n_100 as f64/ nq as f64;
    Ok((r_1, r_10, r_100))
}

fn read_csv(file_path: &str) -> Result<Vec<usize>> {
    let mut rdr = Reader::from_path(file_path)?;
    let mut values = Vec::new();

    for result in rdr.records() {
        let value = result?;
        let value:usize = record[0].parse::<usize>()?;
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

    let _ = dataset.initialize(&opts);
    info!("Seed index constructed.");

    // Load GroundTruth

    // Load queries
    let queries = FvecsDataset::new("data/sift_query".to_string()).unwrap();
    let queries = FlattenedVecs::from(&queries);

    // Load predicates
    let predicate = Some(PredicateQuery::new(1));
    let filter_id_map = todo!(); // TODOM: ask Lachlan!
    let latencies = vec![];
    for k in [5, 10, 15, 20]{
        let res = query_loop(dataset, queries, filter_id_map, k);
        match res {
            Ok((lat)) => {latencies.push(average_duration(lat))}
            Err(_) => {println!("Some error in calculating top-k vectors")}
        }
    }

    // Log results to CSV
    let mut wtr = Writer::from_path("output.csv")?;
    writer.write_record(&["Average Latency (s)", "Recall@K"])?;
    for d in latencies {
        writer.write_record(&[d.as_secs().to_string()])?;
    }
    writer.flush()?;

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