use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Error, Filter, Metric, Result, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub dimension: usize,
    pub metric: Metric,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpsertPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: BTreeMap<String, Value>,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    /// A metric-independent score where larger is better.
    pub score: f32,
    pub metadata: BTreeMap<String, Value>,
    pub sequence: u64,
}

#[derive(Debug)]
pub struct Collection {
    config: CollectionConfig,
    points: HashMap<String, Point>,
    sequence: u64,
}

impl Collection {
    pub fn new(config: CollectionConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(Error::InvalidConfig("dimension must be greater than zero"));
        }
        Ok(Self {
            config,
            points: HashMap::new(),
            sequence: 0,
        })
    }

    pub fn config(&self) -> CollectionConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn latest_sequence(&self) -> u64 {
        self.sequence
    }

    /// Atomically validates and applies a batch, returning its commit sequence.
    pub fn upsert(&mut self, points: Vec<UpsertPoint>) -> Result<u64> {
        let sequence = self.next_sequence()?;
        let prepared = self.prepare_upsert(points, sequence)?;
        self.apply_prepared_upsert(prepared, sequence);
        Ok(sequence)
    }

    /// Deletes IDs as one atomic batch. Missing IDs are valid no-ops.
    pub fn delete<'a>(&mut self, ids: impl IntoIterator<Item = &'a str>) -> Result<u64> {
        let sequence = self.next_sequence()?;
        let ids: Vec<&str> = ids.into_iter().collect();
        for id in ids {
            self.points.remove(id);
        }
        self.sequence = sequence;
        Ok(sequence)
    }

    pub fn get(&self, id: &str) -> Option<&Point> {
        self.points.get(id)
    }

    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
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
        let query = self.config.metric.prepare(query)?;
        Ok(self.top_k(
            &query,
            k,
            self.points
                .values()
                .filter(|point| filter.is_none_or(|filter| filter.matches(&point.metadata))),
        ))
    }

    fn next_sequence(&self) -> Result<u64> {
        self.sequence.checked_add(1).ok_or(Error::SequenceOverflow)
    }

    pub(crate) fn prepare_upsert(
        &self,
        points: Vec<UpsertPoint>,
        sequence: u64,
    ) -> Result<Vec<Point>> {
        if points.is_empty() {
            return Err(Error::InvalidQuery("upsert batch must not be empty"));
        }
        let mut prepared = Vec::with_capacity(points.len());
        for point in points {
            if point.vector.len() != self.config.dimension {
                return Err(Error::InvalidDimension {
                    expected: self.config.dimension,
                    actual: point.vector.len(),
                });
            }
            if point.id.is_empty() {
                return Err(Error::InvalidQuery("point ID must not be empty"));
            }
            prepared.push(Point {
                id: point.id,
                vector: self.config.metric.prepare(point.vector)?,
                metadata: point.metadata,
                sequence,
            });
        }
        Ok(prepared)
    }

    pub(crate) fn apply_prepared_upsert(&mut self, points: Vec<Point>, sequence: u64) {
        for point in points {
            self.points.insert(point.id.clone(), point);
        }
        self.sequence = sequence;
    }

    pub(crate) fn apply_delete_at(&mut self, ids: &[String], sequence: u64) {
        for id in ids {
            self.points.remove(id);
        }
        self.sequence = sequence;
    }

    pub(crate) fn snapshot_points(&self) -> Vec<Point> {
        let mut points: Vec<Point> = self.points.values().cloned().collect();
        points.sort_unstable_by(|left, right| left.id.cmp(&right.id));
        points
    }

    pub(crate) fn search_ids(
        &self,
        query: Vec<f32>,
        k: usize,
        ids: &HashSet<String>,
        filter: Option<&Filter>,
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
        let query = self.config.metric.prepare(query)?;
        Ok(self.top_k(
            &query,
            k,
            ids.iter()
                .filter_map(|id| self.points.get(id))
                .filter(|point| filter.is_none_or(|filter| filter.matches(&point.metadata))),
        ))
    }

    fn top_k<'a>(
        &'a self,
        query: &[f32],
        k: usize,
        points: impl Iterator<Item = &'a Point>,
    ) -> Vec<SearchHit> {
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        for point in points {
            let ranked = RankedPoint {
                point,
                score: self.config.metric.score(query, &point.vector),
            };
            if heap.len() < k {
                heap.push(ranked);
            } else if heap.peek().is_some_and(|worst| ranked < *worst) {
                heap.pop();
                heap.push(ranked);
            }
        }
        let mut ranked = heap.into_vec();
        ranked.sort_unstable();
        ranked
            .into_iter()
            .map(|ranked| SearchHit {
                id: ranked.point.id.clone(),
                score: ranked.score,
                metadata: ranked.point.metadata.clone(),
                sequence: ranked.point.sequence,
            })
            .collect()
    }

    pub(crate) fn restore_snapshot(&mut self, points: Vec<Point>, sequence: u64) -> Result<()> {
        if points.iter().any(|point| point.sequence > sequence) {
            return Err(Error::CorruptStorage(
                "point sequence is newer than its segment".into(),
            ));
        }
        self.points.clear();
        for point in points {
            if self.points.insert(point.id.clone(), point).is_some() {
                return Err(Error::CorruptStorage(
                    "segment contains duplicate point IDs".into(),
                ));
            }
        }
        self.sequence = sequence;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RankedPoint<'a> {
    point: &'a Point,
    score: f32,
}

impl PartialEq for RankedPoint<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.point.id == other.point.id
    }
}

impl Eq for RankedPoint<'_> {}

impl PartialOrd for RankedPoint<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedPoint<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.point.id.cmp(&other.point.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: &str, vector: [f32; 2], tenant: &str, price: i64) -> UpsertPoint {
        UpsertPoint {
            id: id.into(),
            vector: vector.into(),
            metadata: BTreeMap::from([
                ("tenant".into(), Value::Keyword(tenant.into())),
                ("price".into(), Value::Integer(price)),
            ]),
        }
    }

    #[test]
    fn cosine_search_normalizes_and_orders_results() {
        let mut collection = Collection::new(CollectionConfig {
            dimension: 2,
            metric: Metric::Cosine,
        })
        .unwrap();
        collection
            .upsert(vec![
                point("north", [0.0, 8.0], "a", 10),
                point("east", [2.0, 0.0], "a", 20),
                point("diagonal", [1.0, 1.0], "b", 30),
            ])
            .unwrap();

        let hits = collection.search(vec![0.0, 3.0], 2, None).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["north", "diagonal"]
        );
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn filters_are_applied_before_ranking() {
        let mut collection = Collection::new(CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        })
        .unwrap();
        collection
            .upsert(vec![
                point("wrong-tenant", [10.0, 0.0], "b", 5),
                point("too-cheap", [9.0, 0.0], "a", 3),
                point("match", [8.0, 0.0], "a", 12),
            ])
            .unwrap();
        let filter = Filter::And(vec![
            Filter::Eq {
                field: "tenant".into(),
                value: Value::Keyword("a".into()),
            },
            Filter::Range {
                field: "price".into(),
                gte: Some(10.0),
                lt: None,
            },
        ]);

        let hits = collection
            .search(vec![1.0, 0.0], 10, Some(&filter))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "match");
    }

    #[test]
    fn invalid_batch_is_atomic() {
        let mut collection = Collection::new(CollectionConfig {
            dimension: 2,
            metric: Metric::Euclidean,
        })
        .unwrap();
        let result = collection.upsert(vec![
            point("valid", [1.0, 2.0], "a", 1),
            UpsertPoint {
                id: "invalid".into(),
                vector: vec![1.0],
                metadata: BTreeMap::new(),
            },
        ]);

        assert!(matches!(result, Err(Error::InvalidDimension { .. })));
        assert!(collection.is_empty());
        assert_eq!(collection.latest_sequence(), 0);
    }

    #[test]
    fn updates_and_deletes_advance_sequence() {
        let mut collection = Collection::new(CollectionConfig {
            dimension: 2,
            metric: Metric::Euclidean,
        })
        .unwrap();
        assert_eq!(
            collection
                .upsert(vec![point("p", [1.0, 1.0], "a", 1)])
                .unwrap(),
            1
        );
        assert_eq!(
            collection
                .upsert(vec![point("p", [2.0, 2.0], "b", 2)])
                .unwrap(),
            2
        );
        assert_eq!(collection.get("p").unwrap().sequence, 2);
        assert_eq!(collection.delete(["p"]).unwrap(), 3);
        assert!(collection.get("p").is_none());
    }

    #[test]
    fn ties_are_stable_by_point_id() {
        let mut collection = Collection::new(CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        })
        .unwrap();
        collection
            .upsert(vec![
                point("b", [1.0, 0.0], "a", 1),
                point("a", [1.0, 0.0], "a", 1),
            ])
            .unwrap();

        let hits = collection.search(vec![1.0, 0.0], 2, None).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}
