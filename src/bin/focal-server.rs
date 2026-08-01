use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use focal_vector::{Database, DatabaseConfig, ServerConfig, load_server_tls, router, serve_local};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_directory = env::var("FOCAL_DATA_DIR").unwrap_or_else(|_| "./data".into());
    let address: SocketAddr = env::var("FOCAL_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8080".into())
        .parse()?;
    let server_config = ServerConfig {
        bearer_token: env::var("FOCAL_TOKEN")
            .ok()
            .filter(|token| !token.is_empty()),
        max_body_bytes: environment_usize("FOCAL_MAX_BODY_BYTES", 16 * 1024 * 1024)?,
        max_batch_points: environment_usize("FOCAL_MAX_BATCH_POINTS", 1_000)?,
        max_k: environment_usize("FOCAL_MAX_K", 1_000)?,
        max_dimension: environment_usize("FOCAL_MAX_DIMENSION", 4_096)?,
        max_ef_search: environment_usize("FOCAL_MAX_EF_SEARCH", 4_096)?,
        max_concurrent_operations: environment_usize("FOCAL_MAX_CONCURRENT_OPERATIONS", 64)?,
    };
    let database = Arc::new(Database::open(data_directory, DatabaseConfig::default())?);
    if let Some(socket) =
        env::var_os(focal_vector_protocol::SOCKET_ENV).filter(|path| !path.is_empty())
    {
        let socket = std::path::PathBuf::from(socket);
        println!("focal-vector listening on unix://{}", socket.display());
        tokio::select! {
            result = serve_local(&socket, database, server_config) => result?,
            _ = shutdown_signal() => {}
        }
        return Ok(());
    }
    let application = router(database, server_config)?;
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
                shutdown_signal().await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(30)));
            });
            println!("focal-vector listening on https://{address}");
            axum_server::bind_rustls(address, tls)
                .handle(handle)
                .serve(application.into_make_service())
                .await?;
        }
        (None, None) => {
            let listener = tokio::net::TcpListener::bind(address).await?;
            println!("focal-vector listening on http://{address}");
            axum::serve(listener, application)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        _ => return Err("FOCAL_TLS_CERT and FOCAL_TLS_KEY must be set together".into()),
    }
    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        eprintln!("failed to listen for shutdown signal: {error}");
    }
}

fn environment_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{name} must be a positive integer").into()),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}
