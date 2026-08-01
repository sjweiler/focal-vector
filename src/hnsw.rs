use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::{Error, Metric, Result};

const MAX_LEVEL: usize = 32;
const GRAPH_MAGIC: &[u8; 4] = b"FVHG";
const GRAPH_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HnswHit {
    pub id: String,
    /// A metric-independent score where larger is better.
    pub score: f32,
}

#[derive(Debug, Clone)]
struct Node {
    id: String,
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    dimension: usize,
    metric: Metric,
    config: HnswConfig,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn build<I>(dimension: usize, metric: Metric, config: HnswConfig, points: I) -> Result<Self>
    where
        I: IntoIterator<Item = (String, Vec<f32>)>,
    {
        validate_config(dimension, config)?;
        let mut index = Self {
            dimension,
            metric,
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
        };
        let mut ids = HashSet::new();
        for (id, vector) in points {
            if id.is_empty() {
                return Err(Error::InvalidQuery("point ID must not be empty"));
            }
            if !ids.insert(id.clone()) {
                return Err(Error::InvalidQuery("HNSW point IDs must be unique"));
            }
            index.insert(id, vector)?;
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn config(&self) -> HnswConfig {
        self.config
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|node| node.id.as_str())
    }

    pub fn search(&self, query: Vec<f32>, k: usize, ef_search: usize) -> Result<Vec<HnswHit>> {
        if k == 0 {
            return Err(Error::InvalidQuery("k must be greater than zero"));
        }
        if ef_search < k {
            return Err(Error::InvalidQuery("ef_search must be at least k"));
        }
        if query.len() != self.dimension {
            return Err(Error::InvalidDimension {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        let query = self.metric.prepare(query)?;
        let Some(mut current) = self.entry_point else {
            return Ok(Vec::new());
        };

        for layer in (1..=self.max_level).rev() {
            current = self.greedy_closest(&query, current, layer);
        }
        let mut candidates = self.search_layer(&query, current, ef_search, 0);
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|index| HnswHit {
                id: self.nodes[index].id.clone(),
                score: self.score(&query, index),
            })
            .collect())
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        output.extend_from_slice(GRAPH_MAGIC);
        output.push(GRAPH_VERSION);
        output.extend_from_slice(&(self.dimension as u64).to_le_bytes());
        output.push(metric_tag(self.metric));
        put_u32(&mut output, self.config.m)?;
        put_u32(&mut output, self.config.ef_construction)?;
        put_u32(&mut output, self.nodes.len())?;
        output.extend_from_slice(
            &self
                .entry_point
                .map(|value| value as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        put_u32(&mut output, self.max_level)?;
        for node in &self.nodes {
            put_bytes(&mut output, node.id.as_bytes())?;
            for value in &node.vector {
                output.extend_from_slice(&value.to_le_bytes());
            }
            put_u32(&mut output, node.neighbors.len())?;
            for layer in &node.neighbors {
                put_u32(&mut output, layer.len())?;
                for &neighbor in layer {
                    put_u32(&mut output, neighbor)?;
                }
            }
        }
        Ok(output)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        let mut decoder = GraphDecoder::new(bytes);
        if decoder.take(4)? != GRAPH_MAGIC || decoder.byte()? != GRAPH_VERSION {
            return Err(Error::CorruptStorage("invalid HNSW graph header".into()));
        }
        let dimension_u64 = decoder.u64()?;
        let dimension = usize::try_from(dimension_u64)
            .map_err(|_| Error::CorruptStorage("HNSW dimension is too large".into()))?;
        let metric = parse_metric(decoder.byte()?)?;
        let config = HnswConfig {
            m: decoder.usize()?,
            ef_construction: decoder.usize()?,
        };
        validate_config(dimension, config)
            .map_err(|error| Error::CorruptStorage(error.to_string()))?;
        let node_count = decoder.usize()?;
        if node_count > 100_000_000 {
            return Err(Error::CorruptStorage(
                "HNSW node count exceeds limit".into(),
            ));
        }
        let entry_raw = decoder.u64()?;
        let entry_point = if entry_raw == u64::MAX {
            None
        } else {
            Some(
                usize::try_from(entry_raw)
                    .map_err(|_| Error::CorruptStorage("HNSW entry point is too large".into()))?,
            )
        };
        let max_level = decoder.usize()?;
        if max_level > MAX_LEVEL {
            return Err(Error::CorruptStorage("HNSW level exceeds limit".into()));
        }

        let mut nodes = Vec::with_capacity(node_count);
        let mut ids = HashSet::with_capacity(node_count);
        for _ in 0..node_count {
            let id = decoder.string()?;
            if id.is_empty() || !ids.insert(id.clone()) {
                return Err(Error::CorruptStorage(
                    "HNSW graph contains an empty or duplicate ID".into(),
                ));
            }
            let mut vector = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let value = decoder.f32()?;
                if !value.is_finite() {
                    return Err(Error::CorruptStorage(
                        "HNSW graph contains a non-finite vector".into(),
                    ));
                }
                vector.push(value);
            }
            let layer_count = decoder.usize()?;
            if layer_count == 0 || layer_count > MAX_LEVEL + 1 {
                return Err(Error::CorruptStorage(
                    "HNSW node has an invalid layer count".into(),
                ));
            }
            let mut neighbors = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let count = decoder.usize()?;
                if count > config.m {
                    return Err(Error::CorruptStorage(
                        "HNSW neighbor list exceeds configured m".into(),
                    ));
                }
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    layer.push(decoder.usize()?);
                }
                neighbors.push(layer);
            }
            nodes.push(Node {
                id,
                vector,
                neighbors,
            });
        }
        decoder.finish()?;

        if node_count == 0 {
            if entry_point.is_some() || max_level != 0 {
                return Err(Error::CorruptStorage(
                    "empty HNSW graph has an entry point".into(),
                ));
            }
        } else {
            let entry = entry_point.ok_or_else(|| {
                Error::CorruptStorage("non-empty HNSW graph has no entry point".into())
            })?;
            if entry >= node_count || nodes[entry].neighbors.len() <= max_level {
                return Err(Error::CorruptStorage("invalid HNSW entry point".into()));
            }
        }
        for (node_index, node) in nodes.iter().enumerate() {
            for layer in &node.neighbors {
                if layer
                    .iter()
                    .any(|neighbor| *neighbor >= node_count || *neighbor == node_index)
                {
                    return Err(Error::CorruptStorage(
                        "HNSW graph contains an invalid neighbor".into(),
                    ));
                }
            }
        }

        Ok(Self {
            dimension,
            metric,
            config,
            nodes,
            entry_point,
            max_level,
        })
    }

    fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::InvalidDimension {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        let vector = self.metric.prepare(vector)?;
        let level = deterministic_level(&id, self.config.m);
        let new_index = self.nodes.len();
        self.nodes.push(Node {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
        });

        let Some(mut current) = self.entry_point else {
            self.entry_point = Some(new_index);
            self.max_level = level;
            return Ok(());
        };

        if self.max_level > level {
            for layer in ((level + 1)..=self.max_level).rev() {
                current = self.greedy_closest(&self.nodes[new_index].vector, current, layer);
            }
        }

        for layer in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(
                &self.nodes[new_index].vector,
                current,
                self.config.ef_construction,
                layer,
            );
            let selected: Vec<usize> = candidates.into_iter().take(self.config.m).collect();
            if let Some(&closest) = selected.first() {
                current = closest;
            }
            for neighbor in selected {
                self.connect(new_index, neighbor, layer);
            }
        }

        if level > self.max_level {
            self.entry_point = Some(new_index);
            self.max_level = level;
        }
        Ok(())
    }

    fn connect(&mut self, left: usize, right: usize, layer: usize) {
        self.nodes[left].neighbors[layer].push(right);
        self.nodes[right].neighbors[layer].push(left);
        self.prune(left, layer);
        self.prune(right, layer);
    }

    fn prune(&mut self, node: usize, layer: usize) {
        if self.nodes[node].neighbors[layer].len() <= self.config.m {
            return;
        }
        let vector = self.nodes[node].vector.clone();
        let mut neighbors = self.nodes[node].neighbors[layer].clone();
        neighbors.sort_unstable_by(|left, right| {
            compare_scored(
                self.metric.score(&vector, &self.nodes[*left].vector),
                &self.nodes[*left].id,
                self.metric.score(&vector, &self.nodes[*right].vector),
                &self.nodes[*right].id,
            )
        });
        neighbors.truncate(self.config.m);
        self.nodes[node].neighbors[layer] = neighbors;
    }

    fn greedy_closest(&self, query: &[f32], start: usize, layer: usize) -> usize {
        let mut current = start;
        loop {
            let mut best = current;
            let mut best_score = self.score(query, current);
            for &neighbor in self.neighbors(current, layer) {
                let score = self.score(query, neighbor);
                if compare_scored(
                    score,
                    &self.nodes[neighbor].id,
                    best_score,
                    &self.nodes[best].id,
                ) == Ordering::Less
                {
                    best = neighbor;
                    best_score = score;
                }
            }
            if best == current {
                return current;
            }
            current = best;
        }
    }

    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<usize> {
        let entry = HeapItem {
            index: entry,
            score: self.score(query, entry),
        };
        let mut visited = HashSet::with_capacity(ef.saturating_mul(self.config.m).min(65_536));
        visited.insert(entry.index);
        let mut frontier = BinaryHeap::from([entry]);
        let mut results = BinaryHeap::from([Reverse(entry)]);

        while let Some(candidate) = frontier.pop() {
            let worst = results.peek().expect("results are non-empty").0;
            if results.len() >= ef && candidate < worst {
                break;
            }

            for &neighbor in self.neighbors(candidate.index, layer) {
                if !visited.insert(neighbor) {
                    continue;
                }
                let item = HeapItem {
                    index: neighbor,
                    score: self.score(query, neighbor),
                };
                let qualifies =
                    results.len() < ef || item > results.peek().expect("results are non-empty").0;
                if qualifies {
                    frontier.push(item);
                    results.push(Reverse(item));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut result: Vec<usize> = results.into_iter().map(|item| item.0.index).collect();
        result.sort_unstable_by(|left, right| self.compare_nodes(query, *left, *right));
        result
    }

    fn neighbors(&self, node: usize, layer: usize) -> &[usize] {
        self.nodes[node]
            .neighbors
            .get(layer)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn score(&self, query: &[f32], node: usize) -> f32 {
        self.metric.score(query, &self.nodes[node].vector)
    }

    fn compare_nodes(&self, query: &[f32], left: usize, right: usize) -> Ordering {
        compare_scored(
            self.score(query, left),
            &self.nodes[left].id,
            self.score(query, right),
            &self.nodes[right].id,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapItem {
    index: usize,
    score: f32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.index == other.index
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn compare_scored(left_score: f32, left_id: &str, right_score: f32, right_id: &str) -> Ordering {
    right_score
        .total_cmp(&left_score)
        .then_with(|| left_id.cmp(right_id))
}

fn validate_config(dimension: usize, config: HnswConfig) -> Result<()> {
    if dimension == 0 {
        return Err(Error::InvalidConfig("dimension must be greater than zero"));
    }
    if config.m < 2 {
        return Err(Error::InvalidConfig("HNSW m must be at least 2"));
    }
    if config.ef_construction < config.m {
        return Err(Error::InvalidConfig(
            "HNSW ef_construction must be at least m",
        ));
    }
    Ok(())
}

fn deterministic_level(id: &str, m: usize) -> usize {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let threshold = (u64::MAX / m as u64).max(1);
    let mut level = 0;
    while level < MAX_LEVEL && hash < threshold {
        level += 1;
        hash = hash.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    level
}

fn metric_tag(metric: Metric) -> u8 {
    match metric {
        Metric::Cosine => 1,
        Metric::DotProduct => 2,
        Metric::Euclidean => 3,
    }
}

fn parse_metric(tag: u8) -> Result<Metric> {
    match tag {
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::DotProduct),
        3 => Ok(Metric::Euclidean),
        _ => Err(Error::CorruptStorage(format!(
            "unknown HNSW metric tag {tag}"
        ))),
    }
}

fn put_u32(output: &mut Vec<u8>, value: usize) -> Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Error::InvalidQuery("HNSW graph value exceeds format limit"))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    put_u32(output, value.len())?;
    output.extend_from_slice(value);
    Ok(())
}

struct GraphDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> GraphDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::CorruptStorage("HNSW graph length overflow".into()))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::CorruptStorage("truncated HNSW graph".into()))?;
        self.position = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed slice"),
        ))
    }

    fn usize(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("fixed slice")) as usize)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn string(&mut self) -> Result<String> {
        let length = self.usize()?;
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| Error::CorruptStorage("HNSW graph contains invalid UTF-8".into()))
    }

    fn finish(self) -> Result<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::CorruptStorage("trailing bytes in HNSW graph".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(count: usize) -> Vec<(String, Vec<f32>)> {
        (0..count)
            .map(|index| {
                let angle = index as f32 * 0.137;
                (
                    format!("point-{index:04}"),
                    vec![angle.cos(), angle.sin(), (angle * 0.31).cos()],
                )
            })
            .collect()
    }

    #[test]
    fn finds_nearest_points_for_all_metrics() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let index = HnswIndex::build(
                2,
                metric,
                HnswConfig {
                    m: 8,
                    ef_construction: 32,
                },
                [
                    ("a".into(), vec![1.0, 0.0]),
                    ("b".into(), vec![0.0, 1.0]),
                    ("c".into(), vec![-1.0, 0.0]),
                ],
            )
            .unwrap();
            let hits = index.search(vec![0.9, 0.1], 2, 8).unwrap();
            assert_eq!(hits[0].id, "a");
        }
    }

    #[test]
    fn recall_is_high_against_exact_search() {
        let points = points(500);
        let index = HnswIndex::build(
            3,
            Metric::Cosine,
            HnswConfig {
                m: 12,
                ef_construction: 80,
            },
            points.clone(),
        )
        .unwrap();
        let mut matches = 0;
        let mut total = 0;
        for query_index in (0..500).step_by(11) {
            let query = points[query_index].1.clone();
            let approximate: HashSet<String> = index
                .search(query.clone(), 10, 64)
                .unwrap()
                .into_iter()
                .map(|hit| hit.id)
                .collect();
            let prepared = Metric::Cosine.prepare(query).unwrap();
            let mut exact = points.clone();
            exact.sort_unstable_by(|left, right| {
                compare_scored(
                    Metric::Cosine
                        .score(&prepared, &Metric::Cosine.prepare(left.1.clone()).unwrap()),
                    &left.0,
                    Metric::Cosine
                        .score(&prepared, &Metric::Cosine.prepare(right.1.clone()).unwrap()),
                    &right.0,
                )
            });
            for (id, _) in exact.iter().take(10) {
                total += 1;
                matches += usize::from(approximate.contains(id));
            }
        }
        let recall = matches as f64 / total as f64;
        assert!(recall >= 0.95, "recall@10 was {recall:.3}");
    }

    #[test]
    fn validates_configuration_and_queries() {
        assert!(
            HnswIndex::build(
                2,
                Metric::Cosine,
                HnswConfig {
                    m: 1,
                    ef_construction: 10
                },
                []
            )
            .is_err()
        );
        let index = HnswIndex::build(2, Metric::Cosine, HnswConfig::default(), []).unwrap();
        assert!(index.search(vec![1.0], 1, 10).is_err());
        assert!(index.search(vec![1.0, 0.0], 10, 9).is_err());
    }

    #[test]
    fn construction_and_ties_are_deterministic() {
        let build = || {
            HnswIndex::build(
                2,
                Metric::DotProduct,
                HnswConfig::default(),
                [("b".into(), vec![1.0, 0.0]), ("a".into(), vec![1.0, 0.0])],
            )
            .unwrap()
        };
        let first = build().search(vec![1.0, 0.0], 2, 16).unwrap();
        let second = build().search(vec![1.0, 0.0], 2, 16).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0].id, "a");
    }

    #[test]
    fn graph_round_trips_through_binary_format() {
        let original = HnswIndex::build(
            3,
            Metric::Cosine,
            HnswConfig {
                m: 8,
                ef_construction: 40,
            },
            points(100),
        )
        .unwrap();
        let restored = HnswIndex::decode(&original.encode().unwrap()).unwrap();
        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.config(), original.config());
        assert_eq!(
            restored.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap(),
            original.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap()
        );
    }
}
