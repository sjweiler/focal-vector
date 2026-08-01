use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use focal_vector::{
    CollectionConfig, HttpClientTlsConfig, Metric, RaftNode, build_http_client, load_server_tls,
    raft_router,
};
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
        RaftNode::open_with_http_client(
            id,
            directory,
            CollectionConfig {
                dimension,
                metric: metric()?,
            },
            token,
            raft_config,
            build_http_client(client_tls()?.as_ref())?,
        )
        .await?,
    );
    let application = raft_router(Arc::clone(&node));
    let certificate = env::var("FOCAL_TLS_CERT").ok();
    let private_key = env::var("FOCAL_TLS_KEY").ok();
    match (certificate, private_key) {
        (Some(certificate), Some(private_key)) => {
            let client_ca = env::var("FOCAL_TLS_CLIENT_CA").ok();
            let tls = load_server_tls(
                certificate,
                private_key,
                client_ca.as_deref().map(std::path::Path::new),
            )?;
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(30)));
            });
            eprintln!("Focal Vector Raft node {id} listening on https://{bind}");
            axum_server::bind_rustls(bind, tls)
                .handle(handle)
                .serve(application.into_make_service())
                .await?;
        }
        (None, None) => {
            let listener = tokio::net::TcpListener::bind(bind).await?;
            eprintln!(
                "Focal Vector Raft node {id} listening on {}",
                listener.local_addr()?
            );
            axum::serve(listener, application)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
        _ => return Err("FOCAL_TLS_CERT and FOCAL_TLS_KEY must be set together".into()),
    }
    node.raft().shutdown().await?;
    Ok(())
}

fn client_tls() -> Result<Option<HttpClientTlsConfig>, Box<dyn std::error::Error>> {
    let ca_certificate = env::var("FOCAL_TLS_CA").ok().map(Into::into);
    let identity_pem = env::var("FOCAL_TLS_CLIENT_IDENTITY").ok().map(Into::into);
    if ca_certificate.is_none() && identity_pem.is_none() {
        return Ok(None);
    }
    Ok(Some(HttpClientTlsConfig {
        ca_certificate,
        identity_pem,
    }))
}
