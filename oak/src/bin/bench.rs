use anyhow::Result;
use clap::Parser;
use csv::{ReaderBuilder, Writer};
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use oak::bitmask::Bitmask;
use oak::dataset::{OakIndexOptions, SimilaritySearchable, TopKSearchResultBatch};
use oak::fvecs::FvecsDatasetPartition;
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::router::Router;
use slog_scope::{debug, info};
use thiserror::Error;

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

struct QueryStats {
    acorn_recall_10: bool,
    acorn_latency: u128,
    oak_recall_10: bool,
    oak_latency: u128,
    sub_recall_10: bool,
    sub_latency: u128,
}
struct ExperimentResults {
    acorn_latency: f64,
    oak_latency: f64,
    sub_latency: f64,
    acorn_qps: f64,
    oak_qps: f64,
    sub_qps: f64,
    acorn_recall: f64,
    oak_recall: f64,
    sub_recall: f64,
}

fn query_loop(
    dataset: &FvecsDataset,
    subdataset: &FvecsDatasetPartition,
    router: &Router,
    queries: &Vec<FlattenedVecs>,
    bitmask: &Bitmask,
    submask: &Bitmask,
    k: usize,
    gt: &Vec<usize>,
    efsearch: i64,
) -> Result<Vec<QueryStats>> {
    let mut results = vec![];
    for (i, q) in queries.iter().enumerate() {
        // info!("Recall 1: {:?}", gt[i]);

        // First we benchmark ACORN
        let now = tokio::time::Instant::now();
        let result = dataset.search_with_bitmask(q, &bitmask, k, efsearch)?;
        let end = now.elapsed();
        let acorn_latency = end.as_micros();
        let acorn_recall = calculate_recall_1_at_k(gt[i], &result, k)?;

        // Then we direct only to OI showing that searching in a smaller index is more performant

        let sub_now = tokio::time::Instant::now();
        let sub_result = subdataset.search_with_bitmask(q, &submask, k, efsearch)?;
        let sub_end = sub_now.elapsed();
        let sub_latency = sub_end.as_micros();
        let sub_result: Vec<_> = vec![sub_result[0]
            .iter()
            .map(|(index, distance)| {
                (
                    subdataset.original_indices.as_ref().unwrap()[index.clone()] as usize,
                    distance.clone(),
                )
            })
            .collect()];

        let sub_recall = calculate_recall_1_at_k(gt[i], &sub_result, k)?;

        // if !sub_recall {
        //     info!("groundtruth: {:?}", gt[i]);
        //     info!("sub_result: {:?}", sub_result);
        //     info!("result: {:?}", result);
        //     let index = gt[i];

        //     // let attribute = dataset.metadata.attrs[index];
        //     // info!("attribute: {attribute}");
        // }

        // Lastly we integrate OAK's performance
        // We use a router that makes a decision based on performance gain
        // and loss of precision whether to direct it to one of the OIs
        let oak_now = tokio::time::Instant::now();
        let oak_result = router.search_with_bitmask(q, &bitmask, k, efsearch)?;
        let oak_end = oak_now.elapsed();
        let oak_latency = oak_end.as_micros();
        let oak_recall = calculate_recall_1_at_k(gt[i], &oak_result, k)?;

        // We compile these results for a singular query
        results.push(QueryStats {
            acorn_latency,
            acorn_recall_10: acorn_recall,
            sub_latency,
            sub_recall_10: sub_recall,
            oak_latency,
            oak_recall_10: oak_recall,
        });
    }
    Ok(results)
}

fn averages(queries: Vec<QueryStats>) -> Result<ExperimentResults> {
    let acorn_latencies: f64 = queries.iter().map(|qs| qs.acorn_latency).sum::<u128>() as f64;
    let acorn_r10s: f64 = queries.iter().filter(|qs| qs.acorn_recall_10).count() as f64;
    let oak_latencies: f64 = queries.iter().map(|qs| qs.oak_latency).sum::<u128>() as f64;
    let oak_r10s: f64 = queries.iter().filter(|qs| qs.oak_recall_10).count() as f64;
    let sub_latencies: f64 = queries.iter().map(|qs| qs.sub_latency).sum::<u128>() as f64;
    let sub_r10s: f64 = queries.iter().filter(|qs| qs.sub_recall_10).count() as f64;
    let count: f64 = queries.len() as f64;
    Ok(ExperimentResults {
        acorn_latency: acorn_latencies,
        oak_latency: oak_latencies,
        sub_latency: sub_latencies,
        acorn_qps: acorn_latencies / count,
        oak_qps: oak_latencies / count,
        sub_qps: sub_latencies / count,
        acorn_recall: acorn_r10s / count,
        oak_recall: oak_r10s / count,
        sub_recall: sub_r10s / count,
    })
}

// fn validate_recall_1(
//     subdataset: &FvecsDatasetPartition,
//     acorn_result: TopKSearchResultBatch,
// ) -> Result<bool> {
//     let metadata = subdataset.get_metadata();
//
//     for (i, (idx, distance)) in acorn_result[0].iter().enumerate() {
//
//     }
//
//     Ok(true)
// }
fn calculate_recall_1_at_k(
    gt: usize,
    acorn_result: &TopKSearchResultBatch,
    k: usize,
) -> Result<bool> {
    let mut first_value_exists: bool = false;

    for (i, j) in acorn_result[0].iter().enumerate() {
        let index_to_compare = j.0;
        if index_to_compare == gt {
            if i < k {
                first_value_exists = true;
            }
            break;
        }
    }
    Ok(first_value_exists)
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

    let query = PredicateQuery::new(1);

    let mut subdataset = dataset.view(&query);
    let _ = subdataset.initialize(&opts);
    info!("Subindex as view constructed.");

    let dimensionality = dataset.get_dimensionality() as usize;
    // assert_eq!(dimensionality, subdataset.get_dimensionality() as usize);

    let query_set = FvecsDataset::new(args.query, false)?;
    let batched_queries = FlattenedVecs::from(&query_set);
    info!("Query set loaded from disk.");

    let topk = 5;
    let num_queries = batched_queries.len();
    // info!("Total {num_queries} queries loaded");
    let queries = batched_queries.to_vec();

    let mask_main = Bitmask::new(&query, &dataset);
    let mask_sub = Bitmask::new_full(&subdataset);

    // info!("GT loading...");
    // let variable_gt_path = "./outdir/sift_groundtruth.csv";
    let gt = read_csv(args.groundtruth)?;
    // info!("{} gt queries found", gt.len());

    info!("Searching full dataset for {topk} similar vectors for {num_queries} random query , where attr is equal to 1...");
    let efsearch = vec![1, 4, 8, 16, 32];
    // To test ACORN, we simply call search_with_bitmask which routes to the
    // base index for ACORN

    // To test OAK, we use the router which decides whether the query should
    // be redirected to an OI
    let router = Router::new(&dataset, vec![(&mask_main, &subdataset)]);

    let mut results: Vec<ExperimentResults> = vec![];

    for efs in efsearch {
        let qs = query_loop(
            &dataset,
            &subdataset,
            &router,
            &queries,
            &mask_main,
            &mask_sub,
            topk,
            &gt,
            efs,
        )?;
        match averages(qs) {
            Ok(exp_result) => {
                // info!(
                //     "ACORN: QPS was {} microseconds with total latency
                //     being {} for {} and Recall@10 was {}",
                //     exp_result.acorn_qps,
                //     exp_result.acorn_latency,
                //     num_queries,
                //     exp_result.acorn_recall,
                // );
                // info!(
                //     "SUB: QPS was {} queries per microsecond with total latency
                //     as {} and Recall@10 as {}",
                //     exp_result.sub_qps, exp_result.sub_latency, exp_result.sub_recall,
                // );
                // info!(
                //     "OAK: QPS was {} microseconds with total latency
                //     being {} for {} and Recall@10 was {}",
                //     exp_result.oak_qps, exp_result.oak_latency, num_queries, exp_result.oak_recall
                // );
                results.push(exp_result);
            }
            Err(_) => {
                info!("Error calculating averages 2")
            }
        }
    }

    // info!("Got results.");

    let mut wtr = Writer::from_path("experiments.csv")?;

    // Write the header
    wtr.write_record(&[
        "ACORN Latency (microseconds)",
        "ACORN Queries per microsecond",
        "ACORN Recall@10",
        "OAK Latency (microseconds)",
        "OAK Queries per microsecond",
        "OAK Recall@10",
        "SUB Latency (microseconds)",
        "SUB Queries per microsecond",
        "SUB Recall@10",
    ])?;

    // Write the data
    for exp in results.iter() {
        wtr.write_record(&[
            exp.acorn_latency.to_string(),
            exp.acorn_qps.to_string(),
            exp.acorn_recall.to_string(),
            exp.oak_latency.to_string(),
            exp.oak_qps.to_string(),
            exp.oak_recall.to_string(),
            exp.sub_latency.to_string(),
            exp.sub_qps.to_string(),
            exp.sub_recall.to_string(),
        ])?;
    }

    // Flush the writer to ensure all data is written
    wtr.flush()?;
    info!(
        "Data written to experiments.csv. 
    Make sure to copy that into the correct data directory to preserve the 
    results across different experiments."
    );
    Ok(())
}
