use std::cmp::Ordering;
use std::path::Path;
use std::thread;

use crate::{
    Collection, CollectionConfig, Durability, Error, Filter, Result, SearchHit, SharedCollection,
    UpsertPoint,
};

/// A deterministic, statically partitioned collection.
///
/// Point IDs are assigned with FNV-1a instead of `DefaultHasher`, whose output
/// is not a stable storage contract. Changing `shard_count` therefore requires
/// an explicit resharding operation.
#[derive(Debug)]
pub struct ShardedCollection {
    config: CollectionConfig,
    shards: Vec<SharedCollection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardCommit {
    pub shard: usize,
    pub sequence: u64,
    pub affected_points: usize,
}

impl ShardedCollection {
    pub fn open(
        root: impl AsRef<Path>,
        config: CollectionConfig,
        durability: Durability,
        shard_count: usize,
    ) -> Result<Self> {
        if shard_count == 0 {
            return Err(Error::InvalidConfig(
                "shard count must be greater than zero",
            ));
        }
        // Validate the collection config before creating any directories.
        Collection::new(config)?;

        let root = root.as_ref();
        let mut shards = Vec::with_capacity(shard_count);
        for shard in 0..shard_count {
            shards.push(SharedCollection::open(
                root.join(format!("shard-{shard:05}")),
                config,
                durability,
            )?);
        }
        Ok(Self { config, shards })
    }

    pub fn config(&self) -> CollectionConfig {
        self.config
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_for_id(&self, id: &str) -> usize {
        stable_hash(id.as_bytes()) as usize % self.shards.len()
    }

    /// Applies one atomic batch per involved shard.
    ///
    /// The full request is validated before any shard is changed. Storage
    /// failures can still leave a multi-shard request partially committed;
    /// callers that require retry safety should attach an idempotency key at
    /// the replicated-shard layer.
    pub fn upsert(&self, points: Vec<UpsertPoint>) -> Result<Vec<ShardCommit>> {
        if points.is_empty() {
            return Err(Error::InvalidQuery("upsert batch must not be empty"));
        }
        validate_points(self.config, &points)?;

        let mut batches = vec![Vec::new(); self.shards.len()];
        for point in points {
            batches[self.shard_for_id(&point.id)].push(point);
        }
        let mut commits = Vec::new();
        for (shard, points) in batches.into_iter().enumerate() {
            if points.is_empty() {
                continue;
            }
            let affected_points = points.len();
            let sequence = self.shards[shard].upsert(points)?;
            commits.push(ShardCommit {
                shard,
                sequence,
                affected_points,
            });
        }
        Ok(commits)
    }

    pub fn delete(&self, ids: Vec<String>) -> Result<Vec<ShardCommit>> {
        if ids.is_empty() {
            return Err(Error::InvalidQuery("delete batch must not be empty"));
        }
        if ids.iter().any(String::is_empty) {
            return Err(Error::InvalidQuery("point ID must not be empty"));
        }

        let mut batches = vec![Vec::new(); self.shards.len()];
        for id in ids {
            batches[self.shard_for_id(&id)].push(id);
        }
        let mut commits = Vec::new();
        for (shard, ids) in batches.into_iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let affected_points = ids.len();
            let sequence = self.shards[shard].delete(ids)?;
            commits.push(ShardCommit {
                shard,
                sequence,
                affected_points,
            });
        }
        Ok(commits)
    }

    /// Searches every shard concurrently and merges the shard-local top-k
    /// results into an exact global top-k.
    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
        ef_search: usize,
    ) -> Result<Vec<SearchHit>> {
        if k == 0 {
            return Err(Error::InvalidQuery("k must be greater than zero"));
        }
        if query.len() != self.config.dimension {
            return Err(Error::InvalidDimension {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let shard_results = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.shards.len());
            for shard in &self.shards {
                let query = query.clone();
                handles
                    .push(scope.spawn(move || shard.search_with_ef(query, k, filter, ef_search)));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| Error::Concurrency("shard search worker panicked".into()))?
                })
                .collect::<Result<Vec<_>>>()
        })?;

        let mut hits: Vec<_> = shard_results.into_iter().flatten().collect();
        hits.sort_unstable_by(compare_hits);
        hits.truncate(k);
        Ok(hits)
    }

    pub fn len(&self) -> Result<usize> {
        self.shards
            .iter()
            .try_fold(0usize, |total, shard| Ok(total + shard.len()?))
    }

    pub fn is_empty(&self) -> Result<bool> {
        for shard in &self.shards {
            if !shard.is_empty()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn flush(&self) -> Result<Vec<ShardCommit>> {
        self.shards
            .iter()
            .enumerate()
            .map(|(shard, collection)| {
                Ok(ShardCommit {
                    shard,
                    sequence: collection.flush()?,
                    affected_points: collection.len()?,
                })
            })
            .collect()
    }
}

fn validate_points(config: CollectionConfig, points: &[UpsertPoint]) -> Result<()> {
    let mut validator = Collection::new(config)?;
    validator.upsert(points.to_vec()).map(|_| ())
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn compare_hits(left: &SearchHit, right: &SearchHit) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.id.cmp(&right.id))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{Metric, Value};

    use super::*;

    fn directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-sharding-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn config() -> CollectionConfig {
        CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        }
    }

    fn point(id: &str, vector: [f32; 2]) -> UpsertPoint {
        UpsertPoint {
            id: id.into(),
            vector: vector.into(),
            metadata: BTreeMap::from([("tenant".into(), Value::Keyword("a".into()))]),
        }
    }

    #[test]
    fn routing_is_stable() {
        assert_eq!(stable_hash(b"hello"), 0xa430d84680aabd0b);
        let directory = directory("routing");
        let collection =
            ShardedCollection::open(&directory, config(), Durability::Sync, 7).unwrap();
        assert_eq!(collection.shard_for_id("point-42"), 0);
        drop(collection);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn merges_global_top_k_and_survives_restart() {
        let directory = directory("merge");
        let collection =
            ShardedCollection::open(&directory, config(), Durability::Sync, 4).unwrap();
        collection
            .upsert(vec![
                point("a", [1.0, 0.0]),
                point("b", [4.0, 0.0]),
                point("c", [3.0, 0.0]),
                point("d", [2.0, 0.0]),
            ])
            .unwrap();
        let hits = collection.search(vec![1.0, 0.0], 3, None, 64).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["b", "c", "d"]
        );
        collection.flush().unwrap();
        drop(collection);

        let reopened = ShardedCollection::open(&directory, config(), Durability::Sync, 4).unwrap();
        assert_eq!(reopened.len().unwrap(), 4);
        assert_eq!(
            reopened.search(vec![1.0, 0.0], 1, None, 64).unwrap()[0].id,
            "b"
        );
        drop(reopened);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn validates_complete_request_before_mutating_any_shard() {
        let directory = directory("validation");
        let collection =
            ShardedCollection::open(&directory, config(), Durability::Sync, 3).unwrap();
        let bad = UpsertPoint {
            id: "bad".into(),
            vector: vec![1.0],
            metadata: BTreeMap::new(),
        };
        let error = collection
            .upsert(vec![point("valid", [1.0, 0.0]), bad])
            .unwrap_err();
        assert!(matches!(error, Error::InvalidDimension { .. }));
        assert_eq!(collection.len().unwrap(), 0);
        drop(collection);
        fs::remove_dir_all(directory).unwrap();
    }
}
