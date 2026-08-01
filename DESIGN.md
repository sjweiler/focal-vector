# Focal Vector design

## 1. Scope and assumptions

The system stores dense floating-point embeddings plus JSON-like metadata and
returns nearest neighbors under optional metadata filters. It must support
online inserts, updates, and deletes without stopping queries.

The first version is a single-node storage engine exposed through gRPC and
HTTP. It supports cosine similarity, dot product, and Euclidean distance.
Vectors in a collection have one fixed dimension and scalar type. Multi-vector
documents are represented as multiple points sharing a document ID.

Non-goals for the first release are sparse-vector search, arbitrary joins,
cross-collection transactions, and automatic distributed rebalancing.

## 2. Architecture

```text
 client
   |
 HTTP / gRPC
   |
 request validation, admission control, auth
   |
 collection router
   +--------------------------+
   |                          |
 query executor          write coordinator
   |                          |
 snapshot/catalog       append-only WAL
   |                          |
   +------ active memtable ---+
   |             |
 immutable segments <--- flush / compaction workers
   |       |      |
 vectors  HNSW   metadata indexes
```

The read path is shared-nothing across immutable segments. Each segment can be
searched concurrently; results are merged with a bounded top-k heap. The write
path appends to the WAL before updating an in-memory memtable. When the memtable
reaches its byte or age threshold, it becomes immutable and is flushed as a new
segment. A manifest is atomically replaced only after all segment files are
durable.

This LSM-like organization separates fast writes from optimized read files. It
also makes recovery, snapshots, and compaction explicit.

## 3. Data model

```text
Collection {
  id, name, dimension, metric, scalar_type,
  schema, index_config, latest_sequence
}

Point {
  point_id: u128 or UTF-8 string,
  vector: float32[dimension],
  metadata: map<string, scalar | scalar[]>,
  sequence: u64,
  deleted: bool
}
```

Every mutation receives a monotonically increasing collection sequence. An
update is a new point version; a delete is a tombstone. During merge, only the
newest visible sequence for a point ID is eligible. This avoids in-place index
mutation and gives queries a stable snapshot.

Metadata fields must be declared as `keyword`, `integer`, `float`, `boolean`,
or `timestamp` before they are indexed. Unindexed fields remain returnable but
cannot be used as filters.

## 4. Durable storage

Each collection directory contains:

```text
collections/<collection-id>/
  collection.meta          # version, dimension, metric, checksum
  MANIFEST                 # current segment and durable sequence
  write.wal                # checksummed, length-prefixed mutation records
  segment-<sequence>.fvs   # current immutable snapshot
```

The initial segment implementation uses one portable checksummed snapshot file.
When HNSW and field indexes land, the segment becomes a directory of independently
validated vector, graph, payload, and metadata-index files so large components
can be memory-mapped separately.

Segment files are immutable except for a copy-on-write deletion bitmap. File
headers include magic bytes, a format version, endianness, length, and checksum.
New formats are written only after readers for them have shipped.

### Write acknowledgement

1. Validate the complete batch.
2. Allocate consecutive sequence numbers.
3. Append one framed WAL batch with CRC32C.
4. `fdatasync` according to the selected durability mode.
5. Apply the batch atomically to the memtable.
6. Return the highest committed sequence.

Default durability is group commit every 1–5 ms. A `sync=true` request forces
the current group to durable storage. `async` mode may be offered explicitly,
but must never be the default.

Recovery replays valid WAL frames after the manifest's durable sequence and
truncates only an incomplete final frame. A checksum error in the middle of a
log is corruption and must stop recovery rather than silently lose data.

## 5. Index strategy

### Baseline: exact search

Always retain a SIMD-friendly brute-force implementation. It is the correctness
oracle, the best plan for small candidate sets, and the benchmark baseline.
Store vectors contiguously and align rows for efficient dot-product kernels.

For cosine distance, normalize vectors once during ingestion and use dot
product during search. Reject zero-norm vectors for cosine collections.

### Primary ANN index: HNSW

Build one HNSW graph per immutable segment. Recommended starting values are:

```text
M = 16
ef_construction = 200
ef_search = max(64, 4 * requested_k)
```

These are tuning defaults, not constants. Larger `M` improves recall but raises
memory and build cost. `ef_search` is a per-query control and should be bounded
by collection policy to protect tail latency.

The Rust HNSW implementation owns its graph representation and uses deterministic
level selection and tie-breaking, making builds reproducible. Vectors occupy a
contiguous `f32` arena rather than one allocation per node. Construction creates
a serial seed and then plans fixed-size insertion batches in parallel against a
stable graph snapshot; plans are applied in input order so results do not depend
on Rayon scheduling or worker count. Per-worker generation marks replace
per-insertion visited hash sets. The base layer permits `2 * M` neighbors and
selection applies the HNSW diversity heuristic before bounded batch pruning.

The graph is
versioned and persisted inside the checksummed immutable segment. The serialized
node/vector ordering is unchanged by the contiguous in-memory representation.
Unfiltered
queries merge graph candidates with exact results from IDs changed since the
last flush, preserving ANN performance during incremental writes. Replicated
shards adaptively expand HNSW traversal for broad filters and fall back to exact
metadata candidates when selectivity or graph recall requires it; the
single-node persistent path keeps filtered reads exact.

The mutable memtable uses exact search until it is large enough to justify a
temporary HNSW index. A background builder may rebuild that graph in batches;
queries must remain correct while it is absent.

For memory-constrained collections, add scalar quantization first, keeping
full-precision vectors on disk for reranking. Add product quantization or
disk-oriented graphs only after profiling shows memory is the limiting factor.

## 6. Filtered search

Keyword, boolean, and low-cardinality integer values use compressed bitmaps.
Numeric and timestamp ranges use sorted value/posting blocks. The filter engine
produces a segment-local candidate bitmap and an estimated cardinality.

The current Rust engine implements equivalent in-memory posting sets for scalar
equality and ordered numeric postings for ranges. Boolean combinations, negation,
and dirty-point merging preserve filter correctness. Compressed persisted
bitmaps remain a future memory and startup optimization.

The planner chooses among:

| Estimated matches | Plan |
|---:|---|
| under `max(10*k, 10,000)` | exact distance over matching IDs |
| medium selectivity | HNSW with bitmap admission, then exact rerank |
| broad/no filter | ordinary HNSW, then filter and expand `ef_search` if needed |

Post-filtering alone is not sufficient: selective filters can return too few
results. Every plan must return fewer than `k` only when the snapshot truly has
fewer than `k` matching live points or a caller-specified work budget is hit.

## 7. Query path

1. Validate dimension, `k`, metric options, filter syntax, and limits.
2. Resolve a manifest snapshot. If `min_sequence` was supplied, wait up to the
   request deadline until that sequence is visible.
3. Compile the filter and estimate candidates per segment.
4. Search segments in a bounded worker pool; do not create one task per segment.
5. Rerank an oversampled candidate set using full-precision vectors.
6. Deduplicate point IDs by newest visible sequence and apply tombstones.
7. Merge into a fixed-size max heap, materialize requested fields, and return.

Distance semantics must be stable at the API boundary. Prefer a `score` where
higher is always better, while also returning the raw metric distance when
requested.

## 8. API sketch

```http
PUT /v1/collections/{name}
{
  "dimension": 768,
  "metric": "cosine",
  "index": {"type": "hnsw", "m": 16, "ef_construction": 200},
  "fields": {"tenant": "keyword", "created_at": "timestamp"}
}

POST /v1/collections/{name}/points:upsert
{
  "points": [{"id": "doc-42", "vector": [0.1, 0.2],
              "metadata": {"tenant": "acme"}}],
  "sync": false
}

POST /v1/collections/{name}/query
{
  "vector": [0.1, 0.2],
  "k": 10,
  "filter": {"and": [
    {"field": "tenant", "eq": "acme"},
    {"field": "created_at", "gte": "2026-01-01T00:00:00Z"}
  ]},
  "search": {"ef": 96, "rerank": 40},
  "include": ["metadata"],
  "min_sequence": 1234
}
```

Batch sizes, `k`, filter complexity, payload bytes, and query work all need hard
limits. Return explicit resource-exhausted errors rather than allowing a single
request to destabilize the node.

## 9. Concurrency and consistency

`ShardedCollection` implements static data partitioning with stable FNV-1a ID
hashing. Writes are atomic within each shard after whole-request validation.
Queries fan out concurrently and merge shard-local top-k results by descending
score and ascending point ID. A multi-shard write is not a transaction: retry
safety and node failure tolerance require the replicated-shard protocol in
[`DISTRIBUTED.md`](DISTRIBUTED.md).

Readers pin an immutable manifest generation and do not take collection-wide
locks. Writers serialize sequence assignment and WAL append, then publish a new
memtable view. Flush and compaction publish files through atomic manifest swaps.
Reference counts or epochs defer deletion of obsolete segment files until no
reader can observe them.

The default read is the latest locally visible snapshot. Supplying a committed
`min_sequence` provides read-your-writes. Individual batches are atomic within
one collection. This contract is simpler and faster than pretending to provide
general transactions.

## 10. Compaction

Use size-tiered compaction initially. Trigger it when either a level has too many
segments, tombstones exceed a threshold, or duplicate versions cause measurable
read amplification. Compaction merges live newest versions, rebuilds HNSW and
field indexes, writes and syncs a new segment, atomically updates the manifest,
then retires old files after their readers exit.

Throttle compaction by bytes read/written and CPU time. Query latency takes
priority; otherwise a large rebuild will create severe p99 spikes.

Graph and segment construction now runs from a short read-locked snapshot and
outside the collection lock. Publication takes a short write lock. WAL records
that arrive during construction are retained and replayed over the new segment,
and manifest sequence checks prevent stale builders from replacing newer work.

## 10.1 Backup and restore

Backups first publish a snapshot, then copy the checksummed collection metadata,
manifest, immutable segment, and any concurrent WAL delta under a read lock.
Files are synced into a temporary backup directory before an atomic rename.
Restore copies into a new collection directory and validates the ordinary
metadata, segment, graph, and WAL formats when the restored collection opens.

## 11. Scaling beyond one node

Partition by a stable hash of point ID, with an optional routing key such as
tenant ID to keep filtered queries local. Each shard is a replicated log plus
the single-node engine described above. A coordinator fans a query to shard
replicas and merges their top-k results.

Start with replication factor three and quorum writes only if the product needs
node-failure durability. Sharding changes failure handling, consistency, and
operations substantially, so it should not be introduced merely to improve
single-node throughput.

## 12. Performance and observability

Required metrics include:

- query latency by phase and percentile;
- recall sampled against exact search;
- candidates visited, filtered, and reranked;
- WAL append/group-commit latency and bytes;
- memtable and segment count/bytes;
- HNSW build time and graph memory;
- compaction debt, tombstone ratio, and read/write amplification;
- cache hit rate, queue depth, rejected work, and page faults.

Benchmark with real embedding distributions and real filters. Uniform random
vectors hide cluster and selectivity behavior. Each run records data-set hash,
hardware, build revision, index parameters, ingest state, warm/cold cache, QPS,
latency percentiles, recall@k, and memory/disk use.

The performance gate should be a curve, not a single number: recall@10 versus
p50/p95/p99 latency at increasing concurrency. An optimization that wins QPS by
quietly reducing recall is a regression.

## 13. Correctness and failure tests

- Compare ANN and filtered results to exact search on deterministic corpora.
- Property-test mutations, version resolution, tombstones, and filter logic.
- Kill the process after every write/flush/manifest step and verify recovery.
- Corrupt and truncate WAL and segment files and verify explicit failure.
- Run concurrent query/upsert/delete/compaction tests under race detection.
- Verify stable results around NaN, infinity, zero norm, ties, and duplicate IDs.
- Fuzz API decoders and on-disk readers with strict allocation limits.

## 14. Implementation choices

Rust is a strong fit for the storage engine because it provides predictable
memory use, safe concurrency, SIMD access, and mature async/networking support.
Keep CPU-bound search in a dedicated bounded pool instead of an async runtime's
I/O workers. Use memory mapping only for immutable, validated files; malformed
lengths in disk data must never become unchecked memory access.

Suggested module boundaries:

```text
api           request types, validation, HTTP/gRPC adapters
catalog       collections, manifests, snapshots
wal           record framing, group commit, recovery
memtable      current versions and tombstones
segment       immutable file formats and exact scan
index-hnsw    build, load, and search
filter        schema, indexes, expressions, planner
query         fan-out, reranking, version merge, top-k
compaction    flush, merge, publish, garbage collection
metrics       tracing, counters, histograms
bench         ground truth, workload runner, reports
```

Avoid coupling the persisted format to a third-party in-memory HNSW structure.
Wrap the algorithm behind an internal trait and own the serialized format or
version it explicitly; this preserves upgrade and recovery control.

## 15. Delivery milestones

### Milestone 1: correctness baseline

In-memory collections, exact SIMD search, typed filters, deterministic tests,
and a benchmark that generates exact ground truth.

### Milestone 2: durable engine

WAL, recovery, memtable rotation, immutable exact-search segments, manifest
snapshots, deletes, and fault-injection tests.

### Milestone 3: fast ANN

HNSW build/search, full-precision reranking, recall/latency sweeps, bounded query
execution, and index build metrics.

### Milestone 4: sustained operation

Field indexes, adaptive filtered-search planning, compaction, snapshot/restore,
quotas, backpressure, and long-running mixed-workload tests.

Only then should replication and sharding be specified against measured limits.
