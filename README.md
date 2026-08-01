# Focal Vector

[![Rust](https://img.shields.io/badge/Rust-2024%20edition-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Runtime%20dependencies-10-brightgreen.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/Tests-53%20passing-brightgreen.svg)](#try-it)

Focal Vector is a low-latency, durable vector database with both single-node
and replicated, hash-sharded execution paths.

The proposed architecture, data model, APIs, indexing strategy, and delivery
plan are documented in [DESIGN.md](DESIGN.md).

The implementation is in Rust. It provides an exact-search correctness engine,
persisted HNSW indexing, durable storage, concurrent access, and an HTTP service.

The library also provides static hash sharding through `ShardedCollection`:
point IDs route deterministically with FNV-1a, shard searches run concurrently,
and shard-local results merge into a deterministic global top-k. Shard count is
a storage contract; changing it requires an explicit data migration.

Distributed replication uses OpenRaft with durable CRC32C journals for consensus
and applied vector commands. It supports majority-committed writes, durable
request deduplication, linearizable leader reads, snapshot installation,
authenticated peer RPC, membership changes, and leader failover.
Replicated shard reads use snapshot-backed HNSW plus an exact delta for updates,
inserts, and deletes made since the graph was built. Large dirty deltas rebuild
off-lock and publish only if the collection sequence is still current.
Raft state snapshots and journals carry CRC32C integrity checks and fail closed
on corruption or truncation.

Durable collections are backed by a versioned, CRC32C-protected write-ahead log.
Synchronous commits are acknowledged only after `sync_data`, collection
configuration is persisted and verified on reopen, complete mutations recover
after restart, and a torn final frame is safely discarded.

Calling `PersistentCollection::flush` writes a checksummed immutable snapshot,
atomically publishes its manifest, and checkpoints the WAL. The publication
order is crash-safe, and obsolete snapshots are retired after the new manifest
is durable.

The Rust core also includes a deterministic HNSW implementation. It supports all
three metrics, configurable graph density and construction breadth, and a
per-query `ef_search` recall/latency control. Flushes persist the graph inside
the checksummed segment, and unfiltered persistent queries use it after restart.
Exact search remains the reference engine and the path for metadata-filtered
queries.

Committed mutations are kept as a small exact-search delta and merged with HNSW
results, so writes do not disable the immutable graph. The next flush folds that
delta into a newly built graph.

Immutable snapshots also carry in-memory equality and numeric-range indexes for
metadata query planning. Compound filters search only matching snapshot IDs and
merge them with filtered dirty points.

Segment recovery uses a read-only memory map, avoiding a second whole-file input
buffer while checksums and graph structures are decoded.

Collection directories use operating-system exclusive file locks, preventing a
second process or handle from opening the same collection for writing.

## Try it

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --bin focal-bench -- 10000 128 100 96
```

## Run the server

```bash
FOCAL_DATA_DIR=./data \
FOCAL_BIND=127.0.0.1:8080 \
FOCAL_TOKEN=change-me \
cargo run --release --bin focal-server
```

`FOCAL_TOKEN` is optional, but omitting it disables API authentication. The
server binds to localhost by default. Health, readiness, and Prometheus metrics
are public at `/healthz`, `/readyz`, and `/metrics`.

```bash
curl -X PUT http://127.0.0.1:8080/v1/collections/articles \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"dimension":3,"metric":"cosine"}'

curl -X POST http://127.0.0.1:8080/v1/collections/articles/points/upsert \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"points":[{"id":"doc-1","vector":[0.2,0.4,0.8],"metadata":{"tenant":"acme"}}]}'

curl -X POST http://127.0.0.1:8080/v1/collections/articles/query \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.1,0.3,0.9],"k":10,"filter":{"op":"eq","field":"tenant","value":"acme"}}'
```

Request bodies, batch sizes, result counts, and filter nesting are bounded.
Blocking storage work runs outside Tokio's asynchronous worker threads, and the
server shuts down gracefully on Ctrl-C.

Set `FOCAL_TLS_CERT` and `FOCAL_TLS_KEY` to serve HTTPS directly. Setting
`FOCAL_TLS_CLIENT_CA` additionally requires every client to present a certificate
signed by that CA (mTLS). The same variables work for `focal-raft-node`.

Create and restore an atomic backup:

```bash
curl -X POST \
  http://127.0.0.1:8080/v1/collections/articles/backups/articles-2026-07-31 \
  -H 'Authorization: Bearer change-me'

curl -X POST \
  http://127.0.0.1:8080/v1/backups/articles-2026-07-31/restore/articles-restored \
  -H 'Authorization: Bearer change-me'
```

Backups are written under `<data-dir>/.backups` through a synced temporary
directory and atomic rename. They include the immutable segment and any WAL
delta captured under a consistent collection read lock.

## Run a replicated shard

Start one `focal-raft-node` process per voter with a stable node ID, data
directory, advertised bind address, collection dimension, and shared peer token:

```bash
FOCAL_NODE_ID=1 FOCAL_DATA_DIR=./cluster/node-1 \
FOCAL_BIND=127.0.0.1:8101 FOCAL_DIMENSION=768 \
FOCAL_RAFT_TOKEN=change-me cargo run --release --bin focal-raft-node
```

After all initial voters are listening, initialize the cluster exactly once on
node 1:

```bash
curl -X POST http://127.0.0.1:8101/v1/raft/initialize \
  -H 'x-focal-raft-token: change-me' -H 'content-type: application/json' \
  -d '{"members":{"1":"127.0.0.1:8101","2":"127.0.0.1:8102","3":"127.0.0.1:8103"}}'
```

Peer and administration traffic is token-authenticated. Native TLS uses
`FOCAL_TLS_CERT` and `FOCAL_TLS_KEY`; add `FOCAL_TLS_CLIENT_CA` to require mTLS.
Nodes connecting to private-CA HTTPS peers use `FOCAL_TLS_CA` and, for mTLS,
`FOCAL_TLS_CLIENT_IDENTITY` pointing to a combined client certificate/private-key
PEM. Peer and coordinator addresses must use an explicit `https://` scheme.
Node health and readiness are exposed at `/healthz` and `/readyz`; authenticated
Raft state is available at `/v1/raft/status`.

Applications can construct `DistributedCollection` with one `ReplicaSet` per
shard. It hashes writes to the correct replicated shard, tries replicas until a
leader responds, searches shard leaders concurrently, and merges their results
into a deterministic global top-k. `search_result_with_ef` can also perform
explicit stale follower reads and reports the minimum applied Raft index across
all queried shards.

Consensus purge operations atomically checkpoint the complete log state before
truncating the append journal, preventing unbounded journal growth. Checkpoints
are CRC32C-protected and corruption causes startup to fail closed.

Benchmark a deployed set of replicated shards (semicolons separate shards;
commas separate replicas within a shard):

```bash
FOCAL_SHARDS='https://s0n1,https://s0n2,https://s0n3;https://s1n1,https://s1n2,https://s1n3' \
FOCAL_RAFT_TOKEN=change-me FOCAL_DIMENSION=768 \
cargo run --release --bin focal-distributed-bench
```

The client reports ingest vectors/second and query QPS with p50, p95, and p99
latency. Configure point count, query count, and top-k through
`FOCAL_BENCH_POINTS`, `FOCAL_BENCH_QUERIES`, and `FOCAL_BENCH_K`. Set
`FOCAL_BENCH_BATCH_POINTS` to bound each ingestion request (the default is
1,000). Writes accumulate an exact-search delta and HNSW is built lazily by the
benchmark's warm-up query after ingestion, avoiding repeated graph rebuilds
during bulk loading. The report separates ingestion from index warm-up time and
prints raw-vector memory lower bounds; actual process memory is higher because
of graph, metadata, Raft, allocator, and runtime overhead. Set
`FOCAL_BENCH_EF_SEARCH` to tune HNSW recall versus latency (the default is
`max(16 * k, 256)`). Replicated graphs use `M=32` and `ef_construction=400` to
favor recall. `FOCAL_BENCH_CONCURRENCY` controls concurrent clients and
`FOCAL_BENCH_RECALL_QUERIES` controls exact-reference recall sampling. The
report includes throughput, p50/p95/p99, recall@k, and the minimum applied index.

A representative million-vector run can be launched against a deployed cluster:

```bash
FOCAL_SHARDS='http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103' \
FOCAL_RAFT_TOKEN=change-me FOCAL_BENCH_POINTS=1000000 \
FOCAL_BENCH_QUERIES=1000 \
FOCAL_BENCH_BATCH_POINTS=1000 FOCAL_BENCH_CONCURRENCY=32 \
FOCAL_BENCH_RECALL_QUERIES=20 \
cargo run --release --bin focal-distributed-bench
```

TLS is optional for a local run. Leave all `FOCAL_TLS_*` variables unset and
use `http://127.0.0.1:PORT` addresses in `FOCAL_SHARDS`.

Treat results as hardware- and embedding-specific; the repository does not
claim the 1M-vector SLO until this command is run on the intended deployment.

```rust
use std::collections::BTreeMap;
use focal_vector::{Collection, CollectionConfig, Metric, UpsertPoint};

let mut vectors = Collection::new(CollectionConfig {
    dimension: 3,
    metric: Metric::Cosine,
})?;

vectors.upsert(vec![UpsertPoint {
    id: "doc-1".into(),
    vector: vec![0.2, 0.4, 0.8],
    metadata: BTreeMap::new(),
}])?;

let neighbors = vectors.search(vec![0.1, 0.3, 0.9], 10, None)?;
# Ok::<(), focal_vector::Error>(())
```

For durable storage, open the same directory and configuration on each restart:

```rust
use focal_vector::{CollectionConfig, Durability, Metric, PersistentCollection};

let vectors = PersistentCollection::open(
    "./data/articles",
    CollectionConfig { dimension: 768, metric: Metric::Cosine },
    Durability::Sync,
)?;
# Ok::<(), focal_vector::Error>(())
```

Build and query an approximate index:

```rust
use focal_vector::{HnswConfig, HnswIndex, Metric};

let index = HnswIndex::build(
    3,
    Metric::Cosine,
    HnswConfig { m: 16, ef_construction: 200 },
    [
        ("doc-1".into(), vec![0.2, 0.4, 0.8]),
        ("doc-2".into(), vec![0.7, 0.2, 0.1]),
    ],
)?;
let neighbors = index.search(vec![0.1, 0.3, 0.9], 10, 64)?;
# Ok::<(), focal_vector::Error>(())
```

For concurrent access and automatic checkpointing:

```rust
use std::time::Duration;
use focal_vector::{CollectionConfig, Durability, Metric, SharedCollection};

let vectors = SharedCollection::open(
    "./data/articles",
    CollectionConfig { dimension: 768, metric: Metric::Cosine },
    Durability::Sync,
)?;
let flusher = vectors.start_background_flush(Duration::from_secs(1), 10_000)?;

// Clone `vectors` into request workers. Dropping `flusher` stops and joins its
// managed thread; `flusher.stop()` additionally reports background errors.
# drop(flusher);
# Ok::<(), focal_vector::Error>(())
```

## Initial targets

| Workload | Target |
|---|---:|
| Vector dimensions | 64–4096 |
| Collection size | up to 100M vectors per node |
| Query latency | p95 under 20 ms at 1M vectors, top-10 |
| Recall | at least 0.95 against exact top-10 |
| Write durability | acknowledged writes survive process failure |
| Consistency | read-your-writes when a returned sequence is supplied |

Targets are hypotheses until measured on representative hardware and data.

## Recommended implementation order

1. Exact-search reference engine and benchmark harness.
2. ~~WAL and recovery.~~
3. ~~Mutable state flush and immutable segment files.~~
4. ~~HNSW graph with tunable recall/latency and segment persistence.~~
5. ~~Metadata equality/range indexes and filtered-search planning.~~
6. ~~Background snapshot compaction and operational metrics.~~
7. ~~Replication and sharding.~~
