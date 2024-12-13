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

#[derive(Error, Debug)]
pub enum ExampleError {
    #[error("Generic error")]
    GenericError,
    #[error("Failed to start server: {0}")]
    ServerStartError(String),
}

// #[derive(Parser, Debug)]
// #[command(version, about, long_about = None)]
// struct Args {
//     #[arg(short, long, required(true))]
//     dataset: String,
// }
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
/// time_req measures time taken for the vectorDB call
// fn time_req(
//     dataset: &FvecsDataset,
//     query_vector: &FlattenedVecs,
//     filter_id_map: &Bitmask,
//     k: usize) -> Result<(Duration, Result<TopKSearchResultBatch, SearchableError>)> {
//     let now = tokio::time::Instant::now();
//     let result = dataset.search_with_bitmask(&query_vector, *filter_id_map, k);
//     Ok((now.elapsed(), result))
// }

/// This query loop essentially takes a query load, finds the latency of running
/// the queries one at a time, looks at the total time taken and determines QPS
/// and finally, logs the recall for each query as compared to the groundtruth
/// indices
// fn query_loop (
//     dataset: FvecsDataset,
//     query_vectors: Vec<FlattenedVecs>, // TODOM: Ask Lachlan whether its Vec of or not
//     filter_id_map: Bitmask,
//     k: usize,
//     gt: Vec<usize>
// ) -> Result<Vec<QueryStats>> {
//     let mut benchmark_results = vec![];
//     for (index, (op, gt)) in query_vectors.iter().zip(gt.iter()).enumerate() {
//         match time_req(&dataset, &op, &filter_id_map, k) {
//             Ok((latency, Ok(res))) => {
//                 let recall = calculate_recall_1(gt, res);
//                 match recall {
//                     Ok((r1, r10, r100)) => {
//                         benchmark_results.push(QueryStats{
//                             recall_1 : r1,
//                             recall_10: r10,
//                             recall_100: r100,
//                             latency: latency
//                         });
//                     }
//                     Err(_) => info!("Some error running recall calculation")
//                 }
//             }
//             Ok((_, Err(e))) => {println!("Search operation failed with error {:?}", e)}
//             Err(e) => {println!("Timing operation failed with error {:?}", e)}
//         }
//     }
//     info!("Completed benchmarking queries!");
//     Ok(benchmark_results)
// }

fn averages(queries: Vec<QueryStats>) -> Result<(f32, f32, f32, f32)> {
    let total_latencies:f32 = queries.iter().map(|qs| qs.latency).sum::<Duration>().as_secs() as f32;
    let total_r1:f32 = queries.iter().map(|qs| qs.recall_1).count() as f32;
    let total_r10:f32 = queries.iter().map(|qs| qs.recall_10).count() as f32;
    let total_r100:f32 = queries.iter().map(|qs| qs.recall_100).count() as f32;
    let count:f32 = queries.len() as f32;
    Ok((total_latencies as f32 /count as f32, total_r1 as f32 /count as f32, total_r10 as f32 /count as f32, total_r100 as f32 /count as f32))
}

fn calculate_recall_1(gt: &usize, acorn_result: TopKSearchResultBatch) -> Result<(bool, bool, bool)> {
    // todo!(); // Ask lachlan how to iterate through TopKSearchResbatch
    // Figure out how to represent the Groundtruth and index into it!!
    let mut n_1= false;
    let mut n_10 = false;
    let mut n_100 = false;
    for (i, j) in acorn_result[0].iter().enumerate() {
        if j.0 == *gt {
            if i <1 {
                n_1 = true;
                n_10 = true;
                n_100 = true;
                break
            }
            if i < 10 {
                n_10 = true;
                n_100 = true;
                break
            }
            if i < 100 {
                n_100 = true;
                break
            }
        }
    }
    Ok((n_1, n_10, n_100))
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

    // let args = Args::parse();
    info!("Dataset loading...");
    let dataset_path:String = "../../../scripts/get_sift/sift/sift_base.fvecs".to_string();

    let mut dataset = FvecsDataset::new(dataset_path)?;
    info!("Dataset loaded from disk.");

    let opts = OakIndexOptions {
        gamma: 1,
        m: 32,
        m_beta: 64,
    };

    let _ = dataset.build_index(&opts);
    info!("Seed index constructed.");

    

    // Load GroundTruth
    // let base_vectors_path = "./outdir/sift_base.fvecs";
    // let queries_path = "./outdir/sift_query.fvecs";
    info!("GT loading...");
    let groundtruth_path = "../../../scripts/get_sift/outdir/sift_groundtruth.csv";
    // let variable_gt_path = "./outdir/sift_groundtruth.csv";
    let gt = read_csv(groundtruth_path);

    info!("Queries loading...");
    // Load queries
    let queries = FvecsDataset::new("../../../scripts/get_sift/sift/sift_query.fvecs".to_string()).unwrap();
    let queries = FlattenedVecs::from(&queries);
    info!("Queries loaded from disk");

    // Load predicates
    // let predicate = Some(PredicateQuery::new(1));
    // let filter_id_map = todo!(); // TODOM: ask Lachlan!
    // let latencies = vec![];
    // // TODO: vary efSearch
    // let res = query_loop(dataset, queries, filter_id_map, k);
    // match res {
    //     Ok((qs)) => {latencies.append(averages(qs))}
    //     Err(_) => {println!("Some error in calculating top-k vectors")}
    // }

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