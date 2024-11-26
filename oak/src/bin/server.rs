use dropshot::endpoint;
use dropshot::ApiDescription;
use dropshot::ConfigLogging;
use dropshot::ConfigLoggingLevel;
use dropshot::HttpError;
use dropshot::HttpResponseOk;
use dropshot::RequestContext;
use dropshot::ServerBuilder;
use http::Method;
use schemars::JsonSchema;
use serde::Serialize;

use oak::dataset::{Dataset, OakIndexOptions};
use oak::fvecs::{FlattenedVecs, FvecsDataset};
use oak::predicate::PredicateQuery;
use oak::stubs::generate_random_vector;

use clap::Parser;

use tracing::info;
use tracing_subscriber;

use anyhow::Result;
use thiserror::Error;

// Ensure that only one of FAISS or hnsw_rs is used.
#[cfg(all(feature = "hnsw_faiss", feature = "hnsw_rust"))]
compile_error!(
    "Features `hnsw_faiss` and `hnsw_rust` cannot be enabled at the same time. Please enable only one."
);

#[derive(Error, Debug)]
pub enum ServerError {
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

/// Information about the database.
#[derive(Serialize, JsonSchema, Clone)]
struct OakInfo {
    /// Dimensionality of the vectors that are searchable.
    dimensionality: usize,
}

/// Defines the trait that captures all the methods.
#[dropshot::api_description]
trait OakApi {
    /// The context type used within endpoints.
    type Context;

    /// Fetch a project.
    #[endpoint {
        method = GET,
        path = "/info",
    }]
    async fn oak_get_info(
        rqctx: RequestContext<Self::Context>,
    ) -> Result<HttpResponseOk<OakInfo>, HttpError>;
}

/// An empty type to hold the project server context.
///
/// This type is never constructed, and is purely a way to name
/// the specific server impl.
enum ServerImpl {}

impl OakApi for ServerImpl {
    type Context = OakInfo;

    async fn oak_get_info(
        rqctx: RequestContext<Self::Context>,
    ) -> Result<HttpResponseOk<OakInfo>, HttpError> {
        let info = rqctx.context();
        Ok(HttpResponseOk(OakInfo {
            dimensionality: info.dimensionality,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set up a logger.
    let log = ConfigLogging::StderrTerminal {
        level: ConfigLoggingLevel::Debug,
    }
    .to_logger("oak-logger")
    .map_err(|e| ServerError::ServerStartError(e.to_string()))?;

    // TODO: there should only be one logger. I should rewrite the `info!` and `debug!` macros for
    // slog or find the equivalent.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

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

    let dimensionality = dataset.get_dimensionality() as usize;

    // info!("Constructing random vector to query with {dimensionality} dimensions");
    // let query_vector = FlattenedVecs {
    //     dimensionality,
    //     data: generate_random_vector(dimensionality),
    // };
    // let topk = 10;
    // let num_queries = query_vector.len();
    // info!("Searching {topk} similar vectors for {num_queries} queries...");

    // let query: Option<PredicateQuery> = None;

    // let result = dataset.search(query_vector, query, topk);

    let api = oak_api_mod::api_description::<ServerImpl>().unwrap();
    let info = OakInfo { dimensionality };

    // Start the server.
    let server = ServerBuilder::new(api, info, log)
        .start()
        .map_err(|error| ServerError::ServerStartError(format!("{error}")))?;

    server.await.map_err(anyhow::Error::msg)
}
