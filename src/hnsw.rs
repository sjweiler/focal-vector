use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use rayon::prelude::*;

use crate::{Error, Metric, Result};

const MAX_LEVEL: usize = 32;
const PARALLEL_BUILD_SEED: usize = 4_096;
const PARALLEL_BUILD_BATCH: usize = 512;
const GRAPH_MAGIC: &[u8; 4] = b"FVHG";
const GRAPH_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswConfig {
    /// Maximum degree above the base layer. The base layer uses `2 * m`, as
    /// recommended by the HNSW construction algorithm.
    pub m: usize,
    /// Candidate-list width used while inserting each point.
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
    neighbors: Vec<Vec<usize>>,
}

struct PreparedPoint {
    id: String,
    vector: Vec<f32>,
    level: usize,
}

struct InsertionPlan {
    neighbors: Vec<Vec<HeapItem>>,
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    dimension: usize,
    metric: Metric,
    config: HnswConfig,
    nodes: Vec<Node>,
    vectors: Vec<f32>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn build<I>(dimension: usize, metric: Metric, config: HnswConfig, points: I) -> Result<Self>
    where
        I: IntoIterator<Item = (String, Vec<f32>)>,
    {
        validate_config(dimension, config)?;
        let points: Vec<_> = points.into_iter().collect();
        let capacity = points.len();
        let mut ids = HashSet::with_capacity(capacity);
        for (id, _) in &points {
            if id.is_empty() {
                return Err(Error::InvalidQuery("point ID must not be empty"));
            }
            if !ids.insert(id.as_str()) {
                return Err(Error::InvalidQuery("HNSW point IDs must be unique"));
            }
        }
        drop(ids);
        let prepared: Vec<PreparedPoint> = points
            .into_par_iter()
            .map(|(id, vector)| {
                if vector.len() != dimension {
                    return Err(Error::InvalidDimension {
                        expected: dimension,
                        actual: vector.len(),
                    });
                }
                let vector = metric.prepare(vector)?;
                Ok(PreparedPoint {
                    level: deterministic_level(&id, config.m),
                    id,
                    vector,
                })
            })
            .collect::<Result<_>>()?;
        let mut index = Self {
            dimension,
            metric,
            config,
            nodes: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity.saturating_mul(dimension)),
            entry_point: None,
            max_level: 0,
        };
        let mut workspace = BuildWorkspace::with_capacity(capacity);
        let mut prepared = prepared.into_iter();
        for point in prepared.by_ref().take(PARALLEL_BUILD_SEED) {
            index.insert_prepared(point, &mut workspace);
        }

        let worker_count = rayon::current_num_threads();
        let mut workspaces: Vec<BuildWorkspace> = (0..worker_count)
            .map(|_| BuildWorkspace::with_capacity(capacity))
            .collect();
        loop {
            let batch: Vec<_> = prepared.by_ref().take(PARALLEL_BUILD_BATCH).collect();
            if batch.is_empty() {
                break;
            }
            let chunk_size = batch.len().div_ceil(worker_count);
            let mut plans: Vec<Option<InsertionPlan>> =
                std::iter::repeat_with(|| None).take(batch.len()).collect();
            plans
                .par_chunks_mut(chunk_size)
                .zip(batch.par_chunks(chunk_size))
                .zip(workspaces.par_iter_mut())
                .for_each(|((plans, points), workspace)| {
                    for (plan, point) in plans.iter_mut().zip(points) {
                        *plan = Some(index.plan_insert(point, workspace));
                    }
                });
            for (point, plan) in batch.into_iter().zip(plans) {
                index.apply_insertion(point, plan.expect("every insertion was planned"));
            }
        }
        index.compact_neighbors();
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
            .map(|item| HnswHit {
                id: self.nodes[item.index].id.clone(),
                score: item.score,
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
        for (node_index, node) in self.nodes.iter().enumerate() {
            put_bytes(&mut output, node.id.as_bytes())?;
            for value in self.vector(node_index) {
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
        let vector_capacity = node_count.checked_mul(dimension).ok_or_else(|| {
            Error::CorruptStorage("HNSW vector storage size overflows address space".into())
        })?;
        let mut vectors = Vec::with_capacity(vector_capacity);
        let mut ids = HashSet::with_capacity(node_count);
        for _ in 0..node_count {
            let id = decoder.string()?;
            if id.is_empty() || !ids.insert(id.clone()) {
                return Err(Error::CorruptStorage(
                    "HNSW graph contains an empty or duplicate ID".into(),
                ));
            }
            for _ in 0..dimension {
                let value = decoder.f32()?;
                if !value.is_finite() {
                    return Err(Error::CorruptStorage(
                        "HNSW graph contains a non-finite vector".into(),
                    ));
                }
                vectors.push(value);
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
                if count > max_connections(config, neighbors.len()) {
                    return Err(Error::CorruptStorage(
                        "HNSW neighbor list exceeds its configured layer limit".into(),
                    ));
                }
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    layer.push(decoder.usize()?);
                }
                neighbors.push(layer);
            }
            nodes.push(Node { id, neighbors });
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
            vectors,
            entry_point,
            max_level,
        })
    }

    fn insert_prepared(&mut self, point: PreparedPoint, workspace: &mut BuildWorkspace) {
        if self.entry_point.is_none() {
            let level = point.level;
            self.apply_insertion(
                point,
                InsertionPlan {
                    neighbors: vec![Vec::new(); level + 1],
                },
            );
            return;
        }
        let plan = self.plan_insert(&point, workspace);
        self.apply_insertion(point, plan);
    }

    fn plan_insert(&self, point: &PreparedPoint, workspace: &mut BuildWorkspace) -> InsertionPlan {
        let level = point.level;
        let mut current = self
            .entry_point
            .expect("non-empty graph has an entry point");
        if self.max_level > level {
            for layer in ((level + 1)..=self.max_level).rev() {
                current = self.greedy_closest(&point.vector, current, layer);
            }
        }

        let connected_levels = level.min(self.max_level);
        let mut neighbors = vec![Vec::new(); level + 1];
        for layer in (0..=connected_levels).rev() {
            let candidates = self.search_layer_for_build(
                &point.vector,
                current,
                self.config.ef_construction,
                layer,
                workspace,
            );
            let selected = self.select_neighbors(candidates, max_connections(self.config, layer));
            if let Some(closest) = selected.first() {
                current = closest.index;
            }
            neighbors[layer] = selected;
        }
        InsertionPlan { neighbors }
    }

    fn apply_insertion(&mut self, point: PreparedPoint, plan: InsertionPlan) {
        let PreparedPoint { id, vector, level } = point;
        let new_index = self.nodes.len();
        self.vectors.extend_from_slice(&vector);
        self.nodes.push(Node {
            id,
            neighbors: vec![Vec::new(); level + 1],
        });

        if self.entry_point.is_none() {
            self.entry_point = Some(new_index);
            self.max_level = level;
            return;
        }

        for (layer, selected) in plan.neighbors.into_iter().enumerate() {
            self.nodes[new_index].neighbors[layer].extend(selected.iter().map(|item| item.index));
            for neighbor in selected {
                self.nodes[neighbor.index].neighbors[layer].push(new_index);
                if self.nodes[neighbor.index].neighbors[layer].len()
                    >= max_connections(self.config, layer).saturating_mul(2)
                {
                    self.prune(neighbor.index, layer);
                }
            }
        }

        if level > self.max_level {
            self.entry_point = Some(new_index);
            self.max_level = level;
        }
    }

    fn compact_neighbors(&mut self) {
        for node in 0..self.nodes.len() {
            for layer in 0..self.nodes[node].neighbors.len() {
                self.prune(node, layer);
            }
        }
    }

    fn prune(&mut self, node: usize, layer: usize) {
        let limit = max_connections(self.config, layer);
        if self.nodes[node].neighbors[layer].len() <= limit {
            return;
        }
        let mut candidates: Vec<HeapItem> = self.nodes[node].neighbors[layer]
            .iter()
            .map(|&index| HeapItem {
                index,
                score: self.metric.score(self.vector(node), self.vector(index)),
            })
            .collect();
        self.sort_items(&mut candidates);
        self.nodes[node].neighbors[layer] = self
            .select_neighbors(candidates, limit)
            .into_iter()
            .map(|item| item.index)
            .collect();
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

    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<HeapItem> {
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
        let mut result: Vec<HeapItem> = results.into_iter().map(|item| item.0).collect();
        self.sort_items(&mut result);
        result
    }

    fn search_layer_for_build(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
        workspace: &mut BuildWorkspace,
    ) -> Vec<HeapItem> {
        workspace.begin(self.nodes.len());
        let entry = HeapItem {
            index: entry,
            score: self.score(query, entry),
        };
        workspace.visit(entry.index);
        let mut frontier = BinaryHeap::from([entry]);
        let mut results = BinaryHeap::from([Reverse(entry)]);

        while let Some(candidate) = frontier.pop() {
            let worst = results.peek().expect("results are non-empty").0;
            if results.len() >= ef && candidate < worst {
                break;
            }
            for &neighbor in self.neighbors(candidate.index, layer) {
                if !workspace.visit(neighbor) {
                    continue;
                }
                let item = HeapItem {
                    index: neighbor,
                    score: self.score(query, neighbor),
                };
                if results.len() < ef || item > results.peek().expect("results are non-empty").0 {
                    frontier.push(item);
                    results.push(Reverse(item));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut result: Vec<HeapItem> = results.into_iter().map(|item| item.0).collect();
        self.sort_items(&mut result);
        result
    }

    /// HNSW's diversity heuristic avoids filling every adjacency list with a
    /// tight cluster of near-duplicates. Rejected candidates are used as a
    /// fallback so sparse or highly clustered data still reaches `limit`.
    fn select_neighbors(&self, candidates: Vec<HeapItem>, limit: usize) -> Vec<HeapItem> {
        let mut selected = Vec::with_capacity(limit);
        let mut rejected = Vec::new();
        for candidate in candidates {
            if selected.len() == limit {
                break;
            }
            let diverse = selected.iter().all(|other: &HeapItem| {
                self.metric
                    .score(self.vector(candidate.index), self.vector(other.index))
                    < candidate.score
            });
            if diverse {
                selected.push(candidate);
            } else {
                rejected.push(candidate);
            }
        }
        selected.extend(rejected.into_iter().take(limit - selected.len()));
        selected
    }

    fn sort_items(&self, items: &mut [HeapItem]) {
        items.sort_unstable_by(|left, right| {
            compare_scored(
                left.score,
                &self.nodes[left.index].id,
                right.score,
                &self.nodes[right.index].id,
            )
        });
    }

    fn neighbors(&self, node: usize, layer: usize) -> &[usize] {
        self.nodes[node]
            .neighbors
            .get(layer)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn score(&self, query: &[f32], node: usize) -> f32 {
        self.metric.score(query, self.vector(node))
    }

    fn vector(&self, node: usize) -> &[f32] {
        let start = node * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

struct BuildWorkspace {
    visited: Vec<u32>,
    generation: u32,
}

impl BuildWorkspace {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            visited: Vec::with_capacity(capacity),
            generation: 0,
        }
    }

    fn begin(&mut self, node_count: usize) {
        self.visited.resize(node_count, 0);
        if self.generation == u32::MAX {
            self.visited.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    fn visit(&mut self, node: usize) -> bool {
        if self.visited[node] == self.generation {
            false
        } else {
            self.visited[node] = self.generation;
            true
        }
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

fn max_connections(config: HnswConfig, layer: usize) -> usize {
    if layer == 0 {
        config.m.saturating_mul(2)
    } else {
        config.m
    }
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
    fn parallel_batches_are_deterministic() {
        let points = points(PARALLEL_BUILD_SEED + 128);
        let build = || {
            HnswIndex::build(
                3,
                Metric::Cosine,
                HnswConfig {
                    m: 8,
                    ef_construction: 32,
                },
                points.clone(),
            )
            .unwrap()
        };
        assert_eq!(build().encode().unwrap(), build().encode().unwrap());
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
