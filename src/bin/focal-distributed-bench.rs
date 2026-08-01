use std::collections::BTreeMap;
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use focal_vector::{DistributedCollection, ReplicaSet, UpsertPoint};

fn required(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var(name).map_err(|_| format!("{name} is required").into())
}

fn number(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .parse()?)
}

fn replica_sets(specification: &str) -> Result<Vec<ReplicaSet>, Box<dyn std::error::Error>> {
    let shards: Vec<_> = specification
        .split(';')
        .map(|shard| ReplicaSet {
            addresses: shard
                .split(',')
                .map(str::trim)
                .filter(|address| !address.is_empty())
                .map(str::to_owned)
                .collect(),
        })
        .collect();
    if shards.is_empty() || shards.iter().any(|shard| shard.addresses.is_empty()) {
        return Err(
            "FOCAL_SHARDS must contain semicolon-separated shards and comma-separated replicas"
                .into(),
        );
    }
    Ok(shards)
}

fn vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        vector.push(((state >> 40) as f32 + 1.0) / ((1_u32 << 24) as f32));
    }
    vector
}

fn percentile(latencies: &[Duration], percentile: f64) -> Duration {
    let index = ((latencies.len() - 1) as f64 * percentile).round() as usize;
    latencies[index]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shards = replica_sets(&required("FOCAL_SHARDS")?)?;
    let token = required("FOCAL_RAFT_TOKEN")?;
    let dimension = number("FOCAL_DIMENSION", 128)?;
    let point_count = number("FOCAL_BENCH_POINTS", 10_000)?;
    let query_count = number("FOCAL_BENCH_QUERIES", 100)?;
    let k = number("FOCAL_BENCH_K", 10)?;
    let ef_search = number("FOCAL_BENCH_EF_SEARCH", k.saturating_mul(4).max(96))?;
    if dimension == 0 || point_count == 0 || query_count == 0 || k == 0 || ef_search < k {
        return Err("benchmark dimensions and counts must be positive".into());
    }

    let collection = DistributedCollection::new(shards, token)?;
    let points: Vec<_> = (0..point_count)
        .map(|index| UpsertPoint {
            id: format!("bench-{index:012}"),
            vector: vector(index as u64 + 1, dimension),
            metadata: BTreeMap::new(),
        })
        .collect();
    let request_id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
    let ingest_start = Instant::now();
    collection
        .upsert("focal-distributed-bench", request_id, points)
        .await?;
    let ingest_elapsed = ingest_start.elapsed();

    let mut latencies = Vec::with_capacity(query_count);
    let query_start = Instant::now();
    for index in 0..query_count {
        let query = vector(point_count as u64 + index as u64 + 1, dimension);
        let started = Instant::now();
        let hits = collection.search_with_ef(query, k, None, ef_search).await?;
        if hits.is_empty() {
            return Err("query unexpectedly returned no results".into());
        }
        latencies.push(started.elapsed());
    }
    let query_elapsed = query_start.elapsed();
    latencies.sort_unstable();

    println!("shards={}", collection.shard_count());
    println!(
        "points={point_count} dimension={dimension} queries={query_count} k={k} ef_search={ef_search}"
    );
    println!(
        "ingest_seconds={:.3} ingest_vectors_per_second={:.1}",
        ingest_elapsed.as_secs_f64(),
        point_count as f64 / ingest_elapsed.as_secs_f64()
    );
    println!(
        "query_qps={:.1} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3}",
        query_count as f64 / query_elapsed.as_secs_f64(),
        percentile(&latencies, 0.50).as_secs_f64() * 1000.0,
        percentile(&latencies, 0.95).as_secs_f64() * 1000.0,
        percentile(&latencies, 0.99).as_secs_f64() * 1000.0,
    );
    Ok(())
}
