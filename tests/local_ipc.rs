use std::collections::BTreeMap;
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use focal_vector::{Database, DatabaseConfig, ServerConfig, serve_local};
use focal_vector_client::{Client, Metric, Point};

fn test_root() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "focal-vector-local-ipc-{}-{nonce}",
        std::process::id()
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_client_round_trips_core_operations() {
    let root = test_root();
    let data = root.join("data");
    let socket = root.join("runtime").join("focal-vector.sock");
    let database = std::sync::Arc::new(Database::open(&data, DatabaseConfig::default()).unwrap());
    let server_socket = socket.clone();
    let server =
        tokio::spawn(
            async move { serve_local(server_socket, database, ServerConfig::default()).await },
        );

    for _ in 0..100 {
        if socket.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(socket.exists(), "local IPC socket did not become ready");

    let client = Client::new(&socket);
    assert_eq!(client.hello().unwrap(), env!("CARGO_PKG_VERSION"));
    let created = client
        .create_collection("memories", 2, Metric::Cosine)
        .unwrap();
    assert_eq!(created.name, "memories");
    assert_eq!(created.dimension, 2);

    let sequence = client
        .upsert(
            "memories",
            vec![
                Point {
                    id: "one".into(),
                    vector: vec![1.0, 0.0],
                    metadata: BTreeMap::from([(
                        "text".into(),
                        serde_json::Value::String("first".into()),
                    )]),
                },
                Point {
                    id: "two".into(),
                    vector: vec![0.0, 1.0],
                    metadata: BTreeMap::new(),
                },
            ],
        )
        .unwrap();
    assert_eq!(sequence, 1);

    let hits = client
        .query("memories", vec![0.9, 0.1], 1, None, None)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "one");
    assert_eq!(hits[0].metadata["text"], "first");

    assert_eq!(client.delete("memories", vec!["one".into()]).unwrap(), 2);
    assert_eq!(client.list_collections().unwrap()[0].points, 1);

    server.abort();
    let _ = server.await;
    drop(client);
    fs::remove_dir_all(root).unwrap();
}
