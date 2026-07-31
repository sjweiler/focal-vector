# Focal Vector

[![Rust](https://img.shields.io/badge/Rust-2024%20edition-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-0-brightgreen.svg)](Cargo.toml)
[![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen.svg)](#try-it)

Focal Vector is a design for a low-latency, durable vector database. The first
release is deliberately single-node: it optimizes the data path before adding
distributed coordination.

The proposed architecture, data model, APIs, indexing strategy, and delivery
plan are documented in [DESIGN.md](DESIGN.md).

The implementation is in Rust. Its current first milestone is a dependency-free
exact-search engine that defines the behavior later HNSW and persistent storage
layers must match.

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

## Try it

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

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
5. Metadata indexes and filtered-search planning.
6. Background compaction, snapshots, and operational metrics.
7. Replication and sharding only after the single-node SLO is repeatable.
