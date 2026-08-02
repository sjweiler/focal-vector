# Focal Vector

[![Rust](https://img.shields.io/badge/Rust-2024%20edition-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Runtime%20dependencies-12-brightgreen.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/Tests-77%20passing-brightgreen.svg)](#try-it)

Focal Vector is a low-latency, durable vector database with both single-node
and replicated, hash-sharded execution paths.

For FocalDesk, the intended deployment is a separate single-node user service.
The compositor does not link or execute vector storage code; `focaldesk-ai`
reaches the daemon through its private local IPC socket.

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

Calling `PersistentCollection::flush` streams a checksummed immutable snapshot
to a temporary file, atomically publishes its manifest, and checkpoints the
WAL. CRC32C is computed while writing, so flush does not assemble a second
segment-sized byte buffer. The publication order is crash-safe.

The Rust core also includes a deterministic HNSW implementation. It supports all
three metrics, configurable graph density and construction breadth, and a
per-query `ef_search` recall/latency control. Flushes persist the graph inside
the checksummed segment, and unfiltered persistent queries use it after restart.
Exact search remains the reference engine and the path for metadata-filtered
queries.

Committed mutations are kept as a small exact-search delta and merged with HNSW
results, so writes do not disable immutable graphs. Routine flushes build a new
HNSW segment only for changed live IDs. Search fans out across at most eight
immutable graph segments and suppresses stale versions and tombstones against
the authoritative snapshot. The ninth flush compacts them into one full graph;
obsolete files are retired only after the replacement manifest is durable.

Immutable snapshots also carry in-memory equality and numeric-range indexes for
metadata query planning. Compound filters search only matching snapshot IDs and
merge them with filtered dirty points.

Segment recovery uses read-only memory maps. Persistent scalar-int8 HNSW code
slices and aligned full-precision point vectors are scored directly from those
maps instead of being copied into heap arenas. The internal mapped-vector type
is separate from the public `Point`, whose `vector` field remains `Vec<f32>` for
source and wire compatibility. `Collection::get` lazily caches an owned public
view only for explicitly fetched IDs; search and reranking stay zero-copy.
Unflushed writes alone use owned authoritative vectors.

Collection directories use operating-system exclusive file locks, preventing a
second process or handle from opening the same collection for writing.

## Try it

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --bin focal-bench -- 10000 128 100 96
cargo run --release --bin focal-persistence-bench -- 100000 128 1000
```

The benchmark arguments are `points dimensions queries ef_search m
ef_construction`. HNSW construction automatically uses the Rayon CPU pool;
set `RAYON_NUM_THREADS` to override its detected core count. A single build can
measure a recall/latency curve without rebuilding the graph:

```bash
FOCAL_BENCH_EF_SWEEP=96,192,384,768 \
cargo run --release --bin focal-bench -- 1000000 128 100 96 32 400
```

Set `FOCAL_BENCH_STORAGE=int8` to benchmark scalar-int8 graph storage, or
`FOCAL_BENCH_STORAGE=both` for an identical-corpus A/B run. The benchmark
defaults to `f32` so historical measurements remain comparable. It also reports
layer, link, ID, and vector-arena sizes for regression profiling.

`focal-persistence-bench` measures a full durable flush followed by a changed-ID
delta flush. Its arguments are `points dimensions delta_points`. On the same
100k × 128 corpus, changing 1,000 points reduced the segment from 83.6 MB to
0.83 MB and flush time from 8.73 s to 121.6 ms. The exact numbers depend on
storage and CPU, but the benchmark also asserts that publication leaves zero
owned authoritative vector bytes.

### HNSW construction and tuning

The builder stores vectors in a contiguous `f32` or scalar-int8 arena and
constructs the graph in deterministic batches. It builds a 4,096-point seed
graph serially,
plans groups of 512 insertions in parallel against a stable graph snapshot, and
then applies those plans in input order. Each Rayon worker reuses generation-
marked visitation storage, avoiding a hash table allocation for every insertion.
Queries draw generation-marked workspaces from a bounded contention-safe pool.
Neighbor selection uses the HNSW diversity heuristic, with `2 * M` connections
on the base layer and `M` above it. The persisted graph format remains compatible
with graphs written before the contiguous in-memory layout was introduced.

After construction, neighbor lists are flattened into `u32` links and compact
ranges, and point IDs live in one ordinal table instead of being duplicated in
graph nodes. Scalar-int8 dot products use runtime-dispatched AVX2 on supported
x86-64 CPUs and a checked portable accumulator elsewhere.

The following results were measured with 32 visible CPU cores on the benchmark's
deterministic one-million-vector, 128-dimensional cosine corpus. Times are means
over 100 queries; these synthetic results are useful for regression testing but
must not replace recall and percentile measurements on production embeddings.

| Profile | Build time | `ef_search` | Recall@10 | Mean query latency |
|---|---:|---:|---:|---:|
| Fast build (`M=16`, `ef_construction=200`) | 153.4 s | 96 | 63.1% | 0.433 ms |
| Fast build (`M=16`, `ef_construction=200`) | 153.4 s | 1536 | 85.7% | 5.357 ms |
| High recall (`M=32`, `ef_construction=400`) | 501.6 s | 96 | 91.8% | 0.699 ms |
| High recall (`M=32`, `ef_construction=400`) | 501.6 s | 384 | 96.0% | 2.430 ms |
| High recall (`M=32`, `ef_construction=400`) | 501.6 s | 768 | 97.7% | 4.827 ms |

Use `M=32`, `ef_construction=400`, and `ef_search=384` as the starting profile
when recall matters at one million vectors. The default `M=16` profile favors
build time and memory; increasing only `ef_search` cannot recover recall that
was lost during graph construction. Larger `M` also increases resident graph
memory and full rebuild time.

### Scalar-int8 graph storage and reranking

Durable single-node and Raft indexes use per-vector symmetric scalar-int8
quantization. Each node stores one signed byte per component plus a scale and
cached squared norm. Graph traversal uses quantized cosine, dot-product, or
Euclidean scores. Queries oversample four times the requested result count and
rerank candidates against the authoritative full-precision vectors, so scores
returned by durable APIs remain full-precision. Filtered exact-search and dirty
delta paths also remain full-precision.

Rebuilds quantize directly from validated snapshot vectors without cloning
another full `f32` arena. Version-1 `f32` graphs remain readable; the next flush
rewrites them using the version-2 representation.

Release-mode measurements on the development i9-13900K produced:

| Corpus | Storage | Vector arena | Build | Query at `ef=384` | Recall@10 |
|---|---|---:|---:|---:|---:|
| 100k × 128 | `f32` | 51.2 MB | 6.96 s | 1.077 ms | 96.3% |
| 100k × 128 | scalar-int8 | 13.6 MB | 3.47 s | 0.316 ms | 95.1% |
| 50k × 768 | `f32` | 153.6 MB | 16.03 s | 3.167 ms | 83.9% |
| 50k × 768 | scalar-int8 | 38.8 MB | 3.85 s | 1.009 ms | 83.3% |

These direct-HNSW recall figures exclude the durable API's full-precision
four-times candidate reranking. Quantization materially reduces rebuild memory,
construction time, and query latency on this AVX2 host. The portable fallback
and other CPUs can have different construction tradeoffs, so both
representations remain available to benchmark and library users.

### Optional CUDA exact search

Focal Vector can keep an immutable full-precision vector snapshot on an NVIDIA
GPU and use cuBLAS for exact unfiltered searches. CUDA remains both a Cargo
feature and a runtime choice; ordinary builds have no CUDA dependency and retain
the CPU/HNSW behavior.

Build and enable automatic selection with:

```bash
cargo build --release --features cuda
FOCAL_CUDA=auto target/release/focal-server
```

The packaged feature targets dynamically loaded CUDA 12 driver and cuBLAS
libraries, so CUDA is not required on the build host.

`FOCAL_CUDA=required` fails collection opening when device or cuBLAS
initialization fails. `FOCAL_CUDA=off` is the default. Select a device with
`FOCAL_CUDA_DEVICE=0` and change the automatic-selection threshold with
`FOCAL_CUDA_MIN_VECTORS=10000`. Library users can instead call
`PersistentCollection::open_with_cuda` or `SharedCollection::open_with_cuda`
with `CudaSearchConfig`.

The CUDA cache is rebuilt after a successful flush. Updates made since that
snapshot remain on the exact CPU delta path, then both top-k lists are merged.
Metadata-filtered searches continue to use the existing metadata index and CPU
exact engine. CUDA failures while refreshing the rebuildable cache disable it
without affecting the durable collection.

The current implementation copies one score per snapshot vector back to the
host for deterministic top-k selection and full-precision CPU reranking. This
is a useful exact-search baseline; a device-side top-k reduction is the next
optimization for very large collections. One million 128-dimensional `f32`
vectors occupy about 488 MiB of VRAM before score and cuBLAS workspaces, so
benchmark against scalar-int8 HNSW on the actual query workload.

## Operational readiness and limits

Focal Vector is ready for daily dogfooding and beta use as FocalDesk's local,
single-user semantic-search sidecar. Run it as a separate process over the Unix
socket so indexing, compaction, and recovery work cannot block the desktop
compositor. Keep source documents in their authoritative store and treat the
vector collection as a durable but rebuildable search index.

Before relying on a deployment, measure recall with its actual embedding model
and query corpus. The default `M=16`, `ef_construction=200` profile favors build
speed. For important 768-dimensional retrieval, test `M=32`,
`ef_construction=400`, and `ef_search=384` against exact top-k results. Exact
reranking corrects candidate order and scores, but cannot recover a relevant
point that approximate traversal never selected.

Recommended local-operation practices:

- Enable periodic backups and verify that they restore before deleting source
  data.
- Bound Rayon threads, concurrent operations, and background-flush frequency so
  interactive FocalDesk work retains CPU and I/O headroom.
- Monitor mapped and owned points, pending IDs, segment count, query percentiles,
  compaction time, disk space, and process RSS.
- Leave temporary space for a complete compacted segment even though routine
  flushes write only changed points and tombstones.
- Benchmark cold starts and cold page-cache behavior; mapped pages consume OS
  page cache even though they do not count as Rust heap allocations.

Do not use Focal Vector as:

- the canonical document store or a replacement for relational transactions,
  joins, and general SQL;
- a safety-critical exact-retrieval system where one missed candidate is
  unacceptable;
- an untrusted public multi-tenant service without additional authentication,
  quotas, isolation, and abuse controls;
- encrypted storage (use filesystem permissions and encrypted disks when data
  at rest is sensitive);
- proven cross-region disaster-recovery infrastructure; Raft support still
  requires deployment-specific partition, failover, and restore testing;
- a validated 100-million-vector solution—the target remains a hypothesis until
  representative multi-million and 100-million-vector tests are completed.

The remaining work is primarily operational hardening rather than another
storage rewrite: long-running mixed read/write soak tests, disk-full and real
power-loss fault injection, multi-million-vector cold-cache benchmarks,
configurable HNSW construction parameters for durable collections, byte-aware
compaction policies, and lower-overhead heap storage for IDs and metadata.

## Run the server

### FocalDesk local sidecar

Set `FOCAL_VECTOR_SOCKET` to disable the TCP listener and expose only the
versioned Unix-socket API. The daemon creates the socket with mode `0600`,
protects its parent directory with mode `0700`, verifies peer credentials, and
accepts only clients running as the same user.

```bash
FOCAL_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/focaldesk/vector" \
FOCAL_VECTOR_SOCKET="$XDG_RUNTIME_DIR/focaldesk/focal-vector.sock" \
RAYON_NUM_THREADS=4 \
FOCAL_MAX_CONCURRENT_OPERATIONS=4 \
cargo run --release --bin focal-server
```

`focal-vector-client` is a small blocking client crate with no dependency on
the storage engine. Async FocalDesk services should call it through
`tokio::task::spawn_blocking`. The client resolves `FOCAL_VECTOR_SOCKET` first
and otherwise uses `$XDG_RUNTIME_DIR/focaldesk/focal-vector.sock`.

```rust,no_run
use focal_vector_client::{Client, Metric};

let client = Client::from_environment()?;
client.hello()?;
if !client.list_collections()?.iter().any(|item| item.name == "memories") {
    client.create_collection("memories", 768, Metric::Cosine)?;
}
# Ok::<(), focal_vector_client::Error>(())
```

For the initial desktop deployment, keep the vector daemon below the
compositor and inference services in scheduling priority. Four Rayon threads
and four admitted storage operations are conservative defaults for the
32-thread development machine. A systemd user unit should additionally set a
memory high-water mark appropriate for the embedding size (24 GiB is a safe
starting point on a 64 GiB host), a hard maximum above it, and `Nice=10`.

The integration boundary is `focaldesk-memory`: replace its current
`sqlite-vec` calls with `focal-vector-client` calls while retaining text and
authoritative relational metadata in SQLite. Store the SQLite memory ID as the
Focal Vector point ID. Treat embeddings and the HNSW index as rebuildable data.
The compositor remains unchanged.

### TCP service

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
