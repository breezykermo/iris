use anyhow::Result;
use clap::Parser;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use futures_util::{
    future::{select, Either},
    stream::{FuturesUnordered, Stream, StreamExt, TryStreamExt},
};
use slog_scope::{debug, info};
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

/// time_req measures time taken for the vectorDB call
async fn time_req(op: Op) -> Result<(Duration,), Report> {
    let now = tokio::time::Instant::now();
    let result = op.exec().await?;
    Ok((now.elapsed(), result))
}

async fn req_loop(
    mut ops: impl Stream<Item = (usize, Op)> + Unpin + Send + 'static,
    done: tokio::sync::watch::Receiver<bool>,
) -> Result<(Vec<Duration>, usize), Report> {
    let mut durs = vec![];
    let mut inflight = FuturesUnordered::new();
    let mut arrv = std::time::Instant::now();

    info!("starting requests");
    loop {
        // first check for a finished request.
        let ops_val = match select(ops.next(), inflight.next()).await {
            Either::Right((Some(Ok(d)), _)) => {
                debug!(inflight = inflight.len(), "request done");
                durs.push(d);
                None
            }
            Either::Right((Some(Err(e)), _)) => return Err(e),
            Either::Right((None, f)) => Some(f.await),
            Either::Left((x, _)) => Some(x),
        };

        // if after the above, something happened in incoming request stream -- either the
        // stream directly yielded, or inflight gave us None and we then waited for a request -- then handle that.
        if let Some(ov) = ops_val {
            match ov {
                Some((remaining_cnt, o)) if remaining_cnt > 0 => {
                    inflight.push(time_req(o));
                    let interarrv = arrv.elapsed();
                    arrv = std::time::Instant::now();
                    debug!(
                        remaining_cnt,
                        inflight = inflight.len(),
                        ?interarrv,
                        "new request"
                    );
                }
                _ => {
                    info!(completed = durs.len(), "finished requests");
                    break;
                }
            }
        }

        // This can't be inside the select because then the else clause would never be
        // triggered.
        if *done.borrow() {
            debug!(completed = durs.len(), "stopping");
            break; // the first client finished. stop.
        }
    }

    Ok::<_, Report>((durs, inflight.len()))
}

/// Generates a stream of Ops to be passed to `req_loop` from a dataset, queries, and filter_id_map.
/// This was taken from -
/// https://github.com/akshayknarayan/burrito/blob/1649718d9acb96c440ee423cf43688ffbeed9a5d/kvstore-ycsb/src/lib.rs#L73
/// - a benchmarking project. It was modified to fit vector operations instead of DB operations
pub fn const_paced_ops_stream(
    dataset: &FvecsDataset,
    queries: Vec<&FlattenedVecs>,
    filter_id_map: &Vec<c_char>,
    k: usize,
    interarrival_micros: u64,
) -> impl Stream<Item = (usize, Op)> {
    let len = queries.len();
    let tkr = SpinTicker::new_const(Duration::from_micros(interarrival_micros))
        .zip(futures_util::stream::iter((0..len).rev()));

    tkr.zip(futures_util::stream::iter(queries))
        .map(move |((_, i), query_vector)| {
            // Each item in the stream will include:
            // (Index, Operation, Filter Map, K-value))
            (
                i,
                Op {
                    dataset,
                    query_vector,
                    filter_id_map,
                    k,
                },
            )
        })
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
