use std::time::Duration;
/**
 * Queries are FVecs file and we specify some k for a k-nearest neighbors search
 * For acorn, the query sets must first filter by some predicate and then do the 
 * k-NN search. 
 * 
 * Workloads:
 * 1) All queries filter on the same predicate -> index pre-made
 * 2) Queries filter on many different predicates -> we only have space for some
 * indices and the rest must be brute-forced
 * 3) Creating an index in the middle?
 */
// Trait for different loads
pub trait Load {
    fn load_info(&self) -> String;
}

// SyncQueries load implementation
pub struct SyncQueries {
    pub num_queries: usize, // e.g., 10k queries
}

pub struct ConstPacedQueries {
    pub num_queries: usize,
    pub predicate: u64, //  What would this predicate be? Index into the metadata?
    pub throughput: u64, // what is the interarrival time.
}
// Is this to load the query set into memory? Vec<FVEC>?
impl Load for SyncQueries {
    fn load_info(&self) -> String {
        format!("{} synchronous queries", self.num_queries)
    }
}
/**
 * This was taken from - https://github.com/akshayknarayan/burrito/blob/1649718d9acb96c440ee423cf43688ffbeed9a5d/kvstore-ycsb/src/lib.rs#L73 -
 * a benchmarking project
 */
pub fn const_paced_ops_stream(
    ops: Vec<Op>,
    interarrival_micros: u64,
    client_id: usize,
) -> impl futures_util::stream::Stream<Item = (usize, Op)> + Send {
    let len = ops.len();
    let tkr = poisson_ticker::SpinTicker::new_const_with_log_id( // Can't track this down but some kind of library func?
        Duration::from_micros(interarrival_micros),
        client_id,
    )
    .zip(futures_util::stream::iter((0..len).rev()));
    tkr.zip(futures_util::stream::iter(ops))
        .map(|((_, i), o)| (i, o))
}

fn run_thread<S, MC, Fut>(
    thread_id: usize,
    done_tx: Arc<tokio::sync::watch::Sender<bool>>,
    done_rx: tokio::sync::watch::Receiver<bool>,
    start: Arc<tokio::sync::Barrier>,
    access_by_client: Vec<(usize, Vec<Op>)>,
    interarrival_micros: u64,
    poisson_arrivals: bool,
    make_client: Arc<MC>,
) -> Result<(Vec<Duration>, usize), Report>
where
    S: bertha::ChunnelConnection<Data = kvstore::Msg> + Send + Sync + 'static,
    MC: Fn(usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<KvClient<S>, Report>>,
{
    if access_by_client.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let mut reqs: FuturesUnordered<_> = access_by_client
            .into_iter()
            .map(|(client_id, ops)| {
                assert!(!ops.is_empty());
                let start = start.clone();
                let done_rx = done_rx.clone();
                let make_client = Arc::clone(&make_client);
                async move {
                    let cl = make_client(client_id)
                        .await
                        .wrap_err("could not make client")?;
                    debug!(?thread_id, ?client_id, "wait for start barrier");
                    start.wait().await;
                    if poisson_arrivals {
                        let ops = poisson_paced_ops_stream(ops, interarrival_micros, client_id);
                        req_loop(cl, ops, done_rx.clone(), client_id).await
                    } else {
                        let ops = const_paced_ops_stream(ops, interarrival_micros, client_id);
                        req_loop(cl, ops, done_rx.clone(), client_id).await
                    }
                }
            })
            .collect();

        // do the accesses until the first client is done.
        let (mut durs, mut remaining_inflight) = reqs
            .try_next()
            .await
            .wrap_err("error driving request loop")?
            .expect("No clients?");
        ensure!(!durs.is_empty(), "No requests finished");
        if !reqs.is_empty() {
            info!(?thread_id, "broadcasting done");
            done_tx
                .send(true)
                .wrap_err("failed to broadcast experiment termination")?;

            // collect all the requests that have completed.
            let rest_durs: Vec<(_, _)> = reqs
                .try_collect()
                .await
                .wrap_err("error driving request loop")?;
            let (rest_durs, rest_left_inflight): (Vec<_>, Vec<_>) =
                rest_durs.into_iter().unzip();
            assert!(!rest_durs.is_empty());
            info!(?thread_id, "all clients reported");
            durs.extend(rest_durs.into_iter().flat_map(|x| x.into_iter()));
            remaining_inflight += rest_left_inflight.into_iter().sum::<usize>();
        }

        Ok((durs, remaining_inflight))
    })
}

let start = Arc::new(tokio::sync::Barrier::new(num_clients));
info!(?interarrival_micros, ?num_clients, "starting requests");
let (done_tx, done_rx) = tokio::sync::watch::channel::<bool>(false);
let done_tx = Arc::new(done_tx);
let make_client = Arc::new(make_client);
let mut threads = Vec::with_capacity(num_threads);
for thread_id in 1..num_threads {
    let done_tx = done_tx.clone();
    let done_rx = done_rx.clone();
    let start = start.clone();
    let access_by_client = std::mem::take(&mut access_by_thread[thread_id]);
    let mc = Arc::clone(&make_client);
    let thread_jh = std::thread::spawn(move || {
        run_thread(
            thread_id,
            done_tx,
            done_rx,
            start,
            access_by_client,
            interarrival_micros,
            poisson_arrivals,
            mc,
        )
    });

    threads.push(thread_jh);
}
