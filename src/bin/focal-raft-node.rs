use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use focal_vector::{CollectionConfig, Metric, RaftNode, raft_router};
use openraft::Config;

fn required(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var(name).map_err(|_| format!("{name} is required").into())
}

fn metric() -> Result<Metric, Box<dyn std::error::Error>> {
    match env::var("FOCAL_METRIC")
        .unwrap_or_else(|_| "cosine".into())
        .as_str()
    {
        "cosine" => Ok(Metric::Cosine),
        "dot" | "dot_product" => Ok(Metric::DotProduct),
        "euclidean" => Ok(Metric::Euclidean),
        value => Err(format!("unsupported FOCAL_METRIC: {value}").into()),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let id = required("FOCAL_NODE_ID")?.parse::<u64>()?;
    let token = required("FOCAL_RAFT_TOKEN")?;
    let dimension = required("FOCAL_DIMENSION")?.parse::<usize>()?;
    let bind = env::var("FOCAL_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8081".into())
        .parse::<SocketAddr>()?;
    let directory =
        PathBuf::from(env::var("FOCAL_DATA_DIR").unwrap_or_else(|_| format!("./data/node-{id}")));
    let raft_config = Config {
        cluster_name: env::var("FOCAL_CLUSTER").unwrap_or_else(|_| "focal-vector".into()),
        ..Default::default()
    };
    let node = Arc::new(
        RaftNode::open(
            id,
            directory,
            CollectionConfig {
                dimension,
                metric: metric()?,
            },
            token,
            raft_config,
        )
        .await?,
    );
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!(
        "Focal Vector Raft node {id} listening on {}",
        listener.local_addr()?
    );
    axum::serve(listener, raft_router(Arc::clone(&node)))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    node.raft().shutdown().await?;
    Ok(())
}
