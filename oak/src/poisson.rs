//! Exponentially distributed timer for your Poisson-arrivals needs.

use core::task::{Context, Poll};
use futures_util::stream::Stream;
use rand_distr::{Distribution, Exp};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use std::time::Instant;
use tracing::trace;

/// Calls `tokio::task::yield_now()` in a loop for each tick.
///
/// # Example
/// ```rust
/// # #[tokio::main]
/// # async fn main() {
/// # use tracing_subscriber::prelude::*; use tracing::info;
/// # let subscriber = tracing_subscriber::fmt().with_test_writer()
/// #    .with_max_level(tracing_subscriber::filter::LevelFilter::TRACE).finish().set_default();
/// let mut t = poisson_ticker::SpinTicker::new(std::time::Duration::from_micros(200));
/// let now = std::time::Instant::now();
/// # info!(?now, "start");
/// for _ in 0usize..250 {
///     (&mut t).await;
/// }
/// let el = now.elapsed();
/// # info!(?el, "end");
/// assert!(el > std::time::Duration::from_millis(40));
/// assert!(el < std::time::Duration::from_millis(60));
/// # }
/// ```
pub struct SpinTicker<T>(
    T,
    Option<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>,
);

impl<T: Timer + Unpin> Future for SpinTicker<T> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        if let None = self.1 {
            self.1 = Some(Box::pin(self.0.wait()));
        }

        futures_util::ready!(self.1.as_mut().unwrap().as_mut().poll(cx));
        self.1 = None;
        Poll::Ready(())
    }
}

impl<T: Timer + Unpin> Stream for SpinTicker<T> {
    type Item = ();

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        self.poll(cx).map(Some)
    }
}

impl SpinTicker<()> {
    pub fn new_poisson(d: Duration) -> SpinTicker<SpinTimer> {
        SpinTicker(SpinTimer::new(d), None)
    }

    pub fn new_poisson_with_log_id(d: Duration, id: usize) -> SpinTicker<SpinTimer> {
        SpinTicker(SpinTimer::new_with_log_id(d, id), None)
    }

    pub fn new_const(d: Duration) -> SpinTicker<ConstTimer> {
        SpinTicker(ConstTimer::new(d), None)
    }

    pub fn new_const_with_log_id(d: Duration, id: usize) -> SpinTicker<ConstTimer> {
        SpinTicker(ConstTimer::new_with_log_id(d, id), None)
    }
}

pub trait Timer {
    fn wait(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>>;
}

pub struct SpinTimer {
    distr: Exp<f64>,
    deficit_ns: Arc<AtomicU64>,
    id: Option<usize>,
}

impl SpinTimer {
    fn new(d: Duration) -> Self {
        Self::new_with_log_id(d, None)
    }

    fn new_with_log_id(d: Duration, id: impl Into<Option<usize>>) -> Self {
        let lambda = 1. / d.as_nanos() as f64;
        let r = Exp::new(lambda).expect("Make exponential distr");
        Self {
            distr: r,
            deficit_ns: Default::default(),
            id: id.into(),
        }
    }
}

impl Timer for SpinTimer {
    fn wait(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        let start = Instant::now();
        let mut rng = rand::thread_rng();
        let next_interarrival_ns = self.distr.sample(&mut rng) as u64;
        if self.deficit_ns.load(Ordering::Acquire) > next_interarrival_ns {
            // load doesn't matter, since we don't care about the read
            let deficit = self
                .deficit_ns
                .fetch_sub(next_interarrival_ns, Ordering::Release);
            trace!(?deficit, "returning immediately from deficit");
            return Box::pin(futures_util::future::ready(()));
        }

        let next_dur = Duration::from_nanos(next_interarrival_ns);
        let next_time = start + next_dur;
        let id = self.id;
        let deficit_ns = Arc::clone(&self.deficit_ns);
        Box::pin(async move {
            while Instant::now() < next_time {
                tokio::task::yield_now().await;
            }

            let elapsed = start.elapsed();
            let elapsed_ns = elapsed.as_nanos() as u64;
            let deficit =
                deficit_ns.fetch_add(elapsed_ns - next_interarrival_ns, Ordering::Release);
            trace!(
                ?id,
                ?elapsed,
                ?deficit,
                sampled_wait_ns = ?next_interarrival_ns,
                "waited"
            );
        })
    }
}

pub struct ConstTimer {
    dur_nanos: usize,
    deficit_ns: Arc<AtomicU64>,
    id: Option<usize>,
}

impl ConstTimer {
    fn new(d: Duration) -> Self {
        Self::new_with_log_id(d, None)
    }

    fn new_with_log_id(d: Duration, id: impl Into<Option<usize>>) -> Self {
        Self {
            dur_nanos: d.as_nanos() as _,
            deficit_ns: Default::default(),
            id: id.into(),
        }
    }
}

impl Timer for ConstTimer {
    fn wait(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> {
        let start = Instant::now();
        let dur = self.dur_nanos as u64;
        if self.deficit_ns.load(Ordering::Acquire) > dur {
            // load doesn't matter, since we don't care about the read
            let deficit = self.deficit_ns.fetch_sub(dur, Ordering::Release);
            trace!(?deficit, "returning immediately from deficit");
            return Box::pin(futures_util::future::ready(()));
        }

        let next_dur = Duration::from_nanos(dur);
        let next_time = start + next_dur;
        let id = self.id;
        let deficit_ns = Arc::clone(&self.deficit_ns);
        Box::pin(async move {
            while Instant::now() < next_time {
                tokio::task::yield_now().await;
            }

            let elapsed = start.elapsed();
            let elapsed_ns = elapsed.as_nanos() as u64;
            let deficit = deficit_ns.fetch_add(elapsed_ns - dur, Ordering::Release);
            trace!(
                ?id,
                ?elapsed,
                ?deficit,
                sampled_wait_ns = ?dur,
                "waited"
            );
        })
    }
}
