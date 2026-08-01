# Distributed execution contract

Focal Vector's implemented `ShardedCollection` is the data-partitioning layer.
The following is the safety contract for replication; it is not a claim that
multi-node consensus is implemented yet.

## Shard placement

- A collection records an immutable shard count and replication factor.
- `FNV1a64(point_id) % shard_count` selects the shard.
- Each shard is an independent Raft group with an odd number of voters.
- Placement changes use Raft joint consensus before removing an old voter.

## Writes

- Only the elected shard leader accepts a mutation.
- Every request carries `(client_id, request_id)` for durable deduplication.
- Success follows majority commit and local state-machine application.
- A batch spanning shards returns one commit receipt per shard. It is not a
  cross-shard transaction; retries are idempotent.

## Reads and queries

- Linearizable reads confirm leadership with a quorum before reading a shard.
- Explicit stale reads may query followers at a reported applied index.
- The coordinator requests top-k from every shard, merges by descending score
  and ascending point ID, and reports the minimum applied index.

## Recovery

- Terms, votes, committed entries, membership, deduplication records, and the
  last-applied log ID survive process and power loss.
- Snapshots contain the collection segment and state-machine metadata and are
  checksummed before atomic installation.
- A node does not serve a shard until snapshot installation and log replay
  reach the leader's committed index.

## Required failure tests

- Kill a leader before and after quorum acknowledgement and verify visibility.
- Partition the old leader and verify it rejects writes after losing quorum.
- Restart every voter and verify term, membership, deduplication, and results.
- Corrupt or truncate logs and snapshots and require fail-closed startup.
- Add and remove voters while writes and queries run.
