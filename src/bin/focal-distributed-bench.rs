use std::collections::{BTreeMap, HashSet};
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use focal_vector::{
    DistributedCollection, HttpClientTlsConfig, Metric, ReplicaSet, UpsertPoint, build_http_client,
};
use tokio::task::JoinSet;

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

fn exact_ids(
    query: &[f32],
    point_count: usize,
    dimension: usize,
    k: usize,
    metric: Metric,
) -> HashSet<String> {
    let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
    let mut scored = Vec::with_capacity(k);
    for index in 0..point_count {
        let point = vector(index as u64 + 1, dimension);
        let score = match metric {
            Metric::DotProduct => query
                .iter()
                .zip(&point)
                .map(|(left, right)| left * right)
                .sum(),
            Metric::Cosine => {
                let point_norm = point.iter().map(|value| value * value).sum::<f32>().sqrt();
                query
                    .iter()
                    .zip(&point)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    / (query_norm * point_norm)
            }
            Metric::Euclidean => -query
                .iter()
                .zip(&point)
                .map(|(left, right)| {
                    let difference = left - right;
                    difference * difference
                })
                .sum::<f32>(),
        };
        if scored.len() < k {
            scored.push((index, score));
            continue;
        }
        let (worst_position, &(worst_id, worst_score)) = scored
            .iter()
            .enumerate()
            .min_by(|(_, (left_id, left_score)), (_, (right_id, right_score))| {
                left_score
                    .total_cmp(right_score)
                    .then_with(|| right_id.cmp(left_id))
            })
            .expect("k is positive");
        if score > worst_score || (score == worst_score && index < worst_id) {
            scored[worst_position] = (index, score);
        }
    }
    scored.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    scored
        .into_iter()
        .take(k)
        .map(|(index, _)| format!("bench-{index:012}"))
        .collect()
}

fn tls_config() -> Result<Option<HttpClientTlsConfig>, Box<dyn std::error::Error>> {
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

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shards = replica_sets(&required("FOCAL_SHARDS")?)?;
    let token = required("FOCAL_RAFT_TOKEN")?;
    let dimension = number("FOCAL_DIMENSION", 128)?;
    let point_count = number("FOCAL_BENCH_POINTS", 10_000)?;
    let batch_points = number("FOCAL_BENCH_BATCH_POINTS", 1_000)?;
    let query_count = number("FOCAL_BENCH_QUERIES", 100)?;
    let k = number("FOCAL_BENCH_K", 10)?;
    let ef_search = number("FOCAL_BENCH_EF_SEARCH", k.saturating_mul(16).max(256))?;
    let concurrency = number("FOCAL_BENCH_CONCURRENCY", 1)?;
    let recall_queries = number("FOCAL_BENCH_RECALL_QUERIES", query_count.min(10))?;
    let metric = metric()?;
    if dimension == 0
        || point_count == 0
        || batch_points == 0
        || query_count == 0
        || k == 0
        || ef_search < k
        || concurrency == 0
        || recall_queries > query_count
    {
        return Err("benchmark dimensions and counts must be positive".into());
    }

    let shard_count = shards.len();
    let replica_count: usize = shards.iter().map(|shard| shard.addresses.len()).sum();
    let collection = DistributedCollection::with_http_client(
        shards,
        token,
        build_http_client(tls_config()?.as_ref())?,
    )?;
    let request_id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros() as u64;
    let ingest_start = Instant::now();
    let batch_count = point_count.div_ceil(batch_points);
    let progress_interval = batch_count.div_ceil(20).max(1);
    for (batch_index, start) in (0..point_count).step_by(batch_points).enumerate() {
        let end = start.saturating_add(batch_points).min(point_count);
        let points = (start..end)
            .map(|index| UpsertPoint {
                id: format!("bench-{index:012}"),
                vector: vector(index as u64 + 1, dimension),
                metadata: BTreeMap::new(),
            })
            .collect();
        collection
            .upsert(
                "focal-distributed-bench",
                request_id
                    .checked_add(batch_index as u64)
                    .ok_or("benchmark request ID overflow")?,
                points,
            )
            .await?;
        if (batch_index + 1) % progress_interval == 0 || end == point_count {
            eprintln!(
                "ingest_progress={end}/{point_count} ({:.1}%) elapsed_seconds={:.1}",
                end as f64 * 100.0 / point_count as f64,
                ingest_start.elapsed().as_secs_f64()
            );
        }
    }
    let ingest_elapsed = ingest_start.elapsed();

    eprintln!("building_or_refreshing_indexes_with_warmup_query");
    let warmup_start = Instant::now();
    let warmup = collection
        .search_result_with_ef(
            vector(point_count as u64 + 1, dimension),
            k,
            None,
            ef_search,
            false,
        )
        .await?;
    if warmup.hits.is_empty() {
        return Err("warm-up query unexpectedly returned no results".into());
    }
    let warmup_elapsed = warmup_start.elapsed();

    let mut latencies = Vec::with_capacity(query_count);
    let mut query_results = Vec::with_capacity(query_count);
    let query_start = Instant::now();
    let mut tasks = JoinSet::new();
    let mut next_query = 0;
    while next_query < query_count || !tasks.is_empty() {
        while next_query < query_count && tasks.len() < concurrency {
            let index = next_query;
            next_query += 1;
            let collection = collection.clone();
            let query = vector(point_count as u64 + index as u64 + 1, dimension);
            tasks.spawn(async move {
                let started = Instant::now();
                let result = collection
                    .search_result_with_ef(query, k, None, ef_search, false)
                    .await?;
                Ok::<_, focal_vector::Error>((index, started.elapsed(), result))
            });
        }
        if let Some(result) = tasks.join_next().await {
            let (index, latency, result) = result??;
            if result.hits.is_empty() {
                return Err("query unexpectedly returned no results".into());
            }
            latencies.push(latency);
            query_results.push((index, result));
        }
    }
    let query_elapsed = query_start.elapsed();
    latencies.sort_unstable();

    println!("shards={}", collection.shard_count());
    println!(
        "points={point_count} dimension={dimension} batch_points={batch_points} batches={batch_count} queries={query_count} k={k} ef_search={ef_search} concurrency={concurrency}"
    );
    let logical_vector_bytes = point_count as u128 * dimension as u128 * 4;
    let replicated_vector_bytes =
        logical_vector_bytes * replica_count as u128 / shard_count as u128;
    println!(
        "replicas={replica_count} raw_vector_gib_lower_bound={:.3} estimated_replicated_raw_vector_gib_lower_bound={:.3}",
        logical_vector_bytes as f64 / 1024_f64.powi(3),
        replicated_vector_bytes as f64 / 1024_f64.powi(3)
    );
    println!(
        "ingest_seconds={:.3} ingest_vectors_per_second={:.1}",
        ingest_elapsed.as_secs_f64(),
        point_count as f64 / ingest_elapsed.as_secs_f64()
    );
    println!("index_warmup_seconds={:.3}", warmup_elapsed.as_secs_f64());
    let mut recalled = 0_usize;
    for (index, result) in query_results
        .iter()
        .filter(|(index, _)| *index < recall_queries)
    {
        let query = vector(point_count as u64 + *index as u64 + 1, dimension);
        let exact = exact_ids(&query, point_count, dimension, k, metric);
        recalled += result
            .hits
            .iter()
            .filter(|hit| exact.contains(&hit.id))
            .count();
    }
    let recall = if recall_queries == 0 {
        0.0
    } else {
        recalled as f64 / (recall_queries * k) as f64
    };
    let min_applied_index = query_results
        .iter()
        .filter_map(|(_, result)| result.min_applied_index)
        .min();
    println!(
        "recall_at_{k}={recall:.4} recall_queries={recall_queries} min_applied_index={}",
        min_applied_index.map_or_else(|| "unknown".into(), |index| index.to_string())
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
