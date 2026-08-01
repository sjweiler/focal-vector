use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::io::Write;
use std::sync::{Arc, Mutex};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::{Error, Metric, Result};

const MAX_LEVEL: usize = 32;
const PARALLEL_BUILD_SEED: usize = 4_096;
const PARALLEL_BUILD_BATCH: usize = 512;
const MAX_POOLED_SEARCH_WORKSPACES: usize = 8;
const GRAPH_MAGIC: &[u8; 4] = b"FVHG";
const GRAPH_VERSION: u8 = 2;
const LEGACY_GRAPH_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswVectorStorage {
    F32,
    ScalarInt8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswStats {
    pub nodes: usize,
    pub layers: usize,
    pub links: usize,
    pub id_bytes: usize,
    /// Heap bytes used by vector components and their lookup metadata. Mapped
    /// code pages are deliberately excluded because residency is OS-managed.
    pub vector_heap_bytes: usize,
    pub vectors_memory_mapped: bool,
}

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
    level: u8,
    range_start: u32,
    neighbors: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy)]
struct LinkRange {
    start: u32,
    end: u32,
}

#[derive(Debug, Clone)]
struct CompactGraph {
    ranges: Vec<LinkRange>,
    links: Vec<u32>,
}

struct PreparedPoint {
    id: String,
    vector: PreparedVector,
    level: usize,
}

enum PreparedVector {
    F32(Vec<f32>),
    ScalarInt8(QuantizedVector),
}

impl PreparedVector {
    fn new(vector: Vec<f32>, storage: HnswVectorStorage) -> Self {
        match storage {
            HnswVectorStorage::F32 => Self::F32(vector),
            HnswVectorStorage::ScalarInt8 => Self::ScalarInt8(QuantizedVector::new(&vector)),
        }
    }

    fn as_ref(&self) -> VectorRef<'_> {
        match self {
            Self::F32(vector) => VectorRef::F32(vector),
            Self::ScalarInt8(vector) => vector.as_ref(),
        }
    }
}

struct QuantizedVector {
    codes: Vec<i8>,
    scale: f32,
    squared_norm: f32,
}

impl QuantizedVector {
    fn new(vector: &[f32]) -> Self {
        let max_abs = vector
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let codes: Vec<i8> = vector
            .iter()
            .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let squared_norm = quantized_squared_norm(&codes, scale);
        Self {
            codes,
            scale,
            squared_norm,
        }
    }

    fn as_ref(&self) -> VectorRef<'_> {
        VectorRef::ScalarInt8 {
            codes: &self.codes,
            scale: self.scale,
            squared_norm: self.squared_norm,
        }
    }
}

#[derive(Clone, Copy)]
enum VectorRef<'a> {
    F32(&'a [f32]),
    ScalarInt8 {
        codes: &'a [i8],
        scale: f32,
        squared_norm: f32,
    },
}

#[derive(Debug, Clone)]
enum VectorArena {
    F32(Vec<f32>),
    ScalarInt8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
        squared_norms: Vec<f32>,
    },
    MappedScalarInt8 {
        bytes: Arc<Mmap>,
        offsets: Vec<u64>,
        scales: Vec<f32>,
        squared_norms: Vec<f32>,
    },
}

impl VectorArena {
    fn with_capacity(storage: HnswVectorStorage, vectors: usize, dimension: usize) -> Self {
        let components = vectors.saturating_mul(dimension);
        match storage {
            HnswVectorStorage::F32 => Self::F32(Vec::with_capacity(components)),
            HnswVectorStorage::ScalarInt8 => Self::ScalarInt8 {
                codes: Vec::with_capacity(components),
                scales: Vec::with_capacity(vectors),
                squared_norms: Vec::with_capacity(vectors),
            },
        }
    }

    fn storage(&self) -> HnswVectorStorage {
        match self {
            Self::F32(_) => HnswVectorStorage::F32,
            Self::ScalarInt8 { .. } | Self::MappedScalarInt8 { .. } => {
                HnswVectorStorage::ScalarInt8
            }
        }
    }

    fn push(&mut self, vector: PreparedVector) {
        match (self, vector) {
            (Self::F32(arena), PreparedVector::F32(vector)) => arena.extend_from_slice(&vector),
            (
                Self::ScalarInt8 {
                    codes,
                    scales,
                    squared_norms,
                },
                PreparedVector::ScalarInt8(vector),
            ) => {
                codes.extend_from_slice(&vector.codes);
                scales.push(vector.scale);
                squared_norms.push(vector.squared_norm);
            }
            _ => unreachable!("prepared vectors match their HNSW arena"),
        }
    }

    fn push_mapped(&mut self, offset: usize, scale: f32, squared_norm: f32) -> Result<()> {
        let Self::MappedScalarInt8 {
            offsets,
            scales,
            squared_norms,
            ..
        } = self
        else {
            return Err(Error::CorruptStorage(
                "mapped vector was decoded into an owned arena".into(),
            ));
        };
        offsets.push(u64::try_from(offset).map_err(|_| {
            Error::CorruptStorage("mapped vector offset exceeds u64 address space".into())
        })?);
        scales.push(scale);
        squared_norms.push(squared_norm);
        Ok(())
    }

    fn get(&self, node: usize, dimension: usize) -> VectorRef<'_> {
        let start = node * dimension;
        match self {
            Self::F32(vectors) => VectorRef::F32(&vectors[start..start + dimension]),
            Self::ScalarInt8 {
                codes,
                scales,
                squared_norms,
            } => VectorRef::ScalarInt8 {
                codes: &codes[start..start + dimension],
                scale: scales[node],
                squared_norm: squared_norms[node],
            },
            Self::MappedScalarInt8 {
                bytes,
                offsets,
                scales,
                squared_norms,
            } => {
                let start = offsets[node] as usize;
                let bytes = &bytes[start..start + dimension];
                // SAFETY: i8 and u8 have identical size/alignment and every bit
                // pattern is valid. The returned slice cannot outlive the map.
                let codes =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) };
                VectorRef::ScalarInt8 {
                    codes,
                    scale: scales[node],
                    squared_norm: squared_norms[node],
                }
            }
        }
    }

    fn component_bytes(&self) -> usize {
        match self {
            Self::F32(vectors) => vectors.len() * size_of::<f32>(),
            Self::ScalarInt8 {
                codes,
                scales,
                squared_norms,
            } => {
                codes.len() * size_of::<i8>()
                    + scales.len() * size_of::<f32>()
                    + squared_norms.len() * size_of::<f32>()
            }
            Self::MappedScalarInt8 {
                offsets,
                scales,
                squared_norms,
                ..
            } => {
                offsets.len() * size_of::<u64>()
                    + scales.len() * size_of::<f32>()
                    + squared_norms.len() * size_of::<f32>()
            }
        }
    }
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
    ids: Vec<Box<str>>,
    vectors: VectorArena,
    entry_point: Option<usize>,
    max_level: usize,
    search_workspaces: Arc<Mutex<Vec<BuildWorkspace>>>,
    compact_graph: Option<CompactGraph>,
}

impl HnswIndex {
    pub fn build<I>(dimension: usize, metric: Metric, config: HnswConfig, points: I) -> Result<Self>
    where
        I: IntoIterator<Item = (String, Vec<f32>)>,
    {
        Self::build_with_storage(dimension, metric, config, HnswVectorStorage::F32, points)
    }

    pub fn build_quantized<I>(
        dimension: usize,
        metric: Metric,
        config: HnswConfig,
        points: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (String, Vec<f32>)>,
    {
        Self::build_with_storage(
            dimension,
            metric,
            config,
            HnswVectorStorage::ScalarInt8,
            points,
        )
    }

    pub fn build_with_storage<I>(
        dimension: usize,
        metric: Metric,
        config: HnswConfig,
        storage: HnswVectorStorage,
        points: I,
    ) -> Result<Self>
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
                let vector = PreparedVector::new(metric.prepare(vector)?, storage);
                Ok(PreparedPoint {
                    level: deterministic_level(&id, config.m),
                    id,
                    vector,
                })
            })
            .collect::<Result<_>>()?;
        Self::build_prepared_points(dimension, metric, config, storage, prepared)
    }

    pub(crate) fn build_quantized_prepared<'a, I>(
        dimension: usize,
        metric: Metric,
        config: HnswConfig,
        points: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (String, &'a [f32])>,
    {
        validate_config(dimension, config)?;
        let points: Vec<_> = points.into_iter().collect();
        let capacity = points.len();
        let mut ids = HashSet::with_capacity(capacity);
        for (id, vector) in &points {
            if id.is_empty() {
                return Err(Error::InvalidQuery("point ID must not be empty"));
            }
            if !ids.insert(id.as_str()) {
                return Err(Error::InvalidQuery("HNSW point IDs must be unique"));
            }
            if vector.len() != dimension {
                return Err(Error::InvalidDimension {
                    expected: dimension,
                    actual: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidVector("components must be finite"));
            }
        }
        let prepared = points
            .into_par_iter()
            .map(|(id, vector)| PreparedPoint {
                level: deterministic_level(&id, config.m),
                id,
                vector: PreparedVector::ScalarInt8(QuantizedVector::new(vector)),
            })
            .collect();
        Self::build_prepared_points(
            dimension,
            metric,
            config,
            HnswVectorStorage::ScalarInt8,
            prepared,
        )
    }

    fn build_prepared_points(
        dimension: usize,
        metric: Metric,
        config: HnswConfig,
        storage: HnswVectorStorage,
        prepared: Vec<PreparedPoint>,
    ) -> Result<Self> {
        let capacity = prepared.len();
        let mut index = Self {
            dimension,
            metric,
            config,
            nodes: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            vectors: VectorArena::with_capacity(storage, capacity, dimension),
            entry_point: None,
            max_level: 0,
            search_workspaces: Arc::new(Mutex::new(Vec::new())),
            compact_graph: None,
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
        index.finalize_compact_graph()?;
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

    pub fn vector_storage(&self) -> HnswVectorStorage {
        self.vectors.storage()
    }

    pub fn vector_storage_bytes(&self) -> usize {
        self.vectors.component_bytes()
    }

    /// Returns true when quantized vector codes are read directly from an
    /// immutable segment mapping instead of a copied heap arena.
    pub fn vectors_are_memory_mapped(&self) -> bool {
        matches!(self.vectors, VectorArena::MappedScalarInt8 { .. })
    }

    pub fn stats(&self) -> HnswStats {
        let (layers, links) = self
            .compact_graph
            .as_ref()
            .map(|graph| (graph.ranges.len(), graph.links.len()))
            .unwrap_or_default();
        HnswStats {
            nodes: self.nodes.len(),
            layers,
            links,
            id_bytes: self.ids.iter().map(|id| id.len()).sum(),
            vector_heap_bytes: self.vectors.component_bytes(),
            vectors_memory_mapped: self.vectors_are_memory_mapped(),
        }
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = &str> {
        self.ids.iter().map(AsRef::as_ref)
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
        let query = PreparedVector::new(self.metric.prepare(query)?, self.vector_storage());
        let query = query.as_ref();
        let Some(mut current) = self.entry_point else {
            return Ok(Vec::new());
        };

        for layer in (1..=self.max_level).rev() {
            current = self.greedy_closest(query, current, layer);
        }
        let mut workspace = self.take_search_workspace();
        let mut candidates = self.search_layer(query, current, ef_search, 0, &mut workspace);
        self.return_search_workspace(workspace);
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|item| HnswHit {
                id: self.ids[item.index].to_string(),
                score: item.score,
            })
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(self.encoded_len()?);
        self.encode_into(&mut output)?;
        Ok(output)
    }

    pub(crate) fn encoded_len(&self) -> Result<usize> {
        let vector_bytes = match self.vector_storage() {
            HnswVectorStorage::F32 => self
                .dimension
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| Error::ResourceExhausted("HNSW encoded size overflows".into()))?,
            HnswVectorStorage::ScalarInt8 => self
                .dimension
                .checked_add(size_of::<f32>())
                .ok_or_else(|| Error::ResourceExhausted("HNSW encoded size overflows".into()))?,
        };
        let mut length = 39_usize;
        for (node_index, id) in self.ids.iter().enumerate() {
            length = length
                .checked_add(4)
                .and_then(|length| length.checked_add(id.len()))
                .and_then(|length| length.checked_add(vector_bytes))
                .and_then(|length| length.checked_add(4))
                .ok_or_else(|| Error::ResourceExhausted("HNSW encoded size overflows".into()))?;
            for layer in 0..=usize::from(self.nodes[node_index].level) {
                length = length
                    .checked_add(4)
                    .and_then(|length| {
                        length.checked_add(self.neighbors(node_index, layer).len().checked_mul(4)?)
                    })
                    .ok_or_else(|| {
                        Error::ResourceExhausted("HNSW encoded size overflows".into())
                    })?;
            }
        }
        Ok(length)
    }

    pub(crate) fn encode_into(&self, output: &mut impl Write) -> Result<()> {
        output.write_all(GRAPH_MAGIC)?;
        output.write_all(&[GRAPH_VERSION])?;
        output.write_all(&(self.dimension as u64).to_le_bytes())?;
        output.write_all(&[metric_tag(self.metric)])?;
        output.write_all(&[storage_tag(self.vector_storage())])?;
        put_u32_writer(output, self.config.m)?;
        put_u32_writer(output, self.config.ef_construction)?;
        put_u32_writer(output, self.nodes.len())?;
        output.write_all(
            &self
                .entry_point
                .map(|value| value as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        )?;
        put_u32_writer(output, self.max_level)?;
        for (node_index, node) in self.nodes.iter().enumerate() {
            put_bytes_writer(output, self.ids[node_index].as_bytes())?;
            match self.vector(node_index) {
                VectorRef::F32(vector) => {
                    for value in vector {
                        output.write_all(&value.to_le_bytes())?;
                    }
                }
                VectorRef::ScalarInt8 {
                    codes,
                    scale,
                    squared_norm: _,
                } => {
                    output.write_all(&scale.to_le_bytes())?;
                    for code in codes {
                        output.write_all(&code.to_le_bytes())?;
                    }
                }
            }
            put_u32_writer(output, usize::from(node.level) + 1)?;
            for layer_index in 0..=usize::from(node.level) {
                let layer = self.neighbors(node_index, layer_index);
                put_u32_writer(output, layer.len())?;
                for &neighbor in layer {
                    put_u32_writer(output, neighbor as usize)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        Self::decode_inner(bytes, None)
    }

    pub(crate) fn decode_mapped(
        bytes: &[u8],
        mapping: Arc<Mmap>,
        mapping_offset: usize,
    ) -> Result<Self> {
        Self::decode_inner(bytes, Some((mapping, mapping_offset)))
    }

    fn decode_inner(bytes: &[u8], mapping: Option<(Arc<Mmap>, usize)>) -> Result<Self> {
        let mut decoder = GraphDecoder::new(bytes);
        if decoder.take(4)? != GRAPH_MAGIC {
            return Err(Error::CorruptStorage("invalid HNSW graph header".into()));
        }
        let version = decoder.byte()?;
        if version != LEGACY_GRAPH_VERSION && version != GRAPH_VERSION {
            return Err(Error::CorruptStorage("invalid HNSW graph header".into()));
        }
        let dimension_u64 = decoder.u64()?;
        let dimension = usize::try_from(dimension_u64)
            .map_err(|_| Error::CorruptStorage("HNSW dimension is too large".into()))?;
        let metric = parse_metric(decoder.byte()?)?;
        let storage = if version == LEGACY_GRAPH_VERSION {
            HnswVectorStorage::F32
        } else {
            parse_storage(decoder.byte()?)?
        };
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
        node_count.checked_mul(dimension).ok_or_else(|| {
            Error::CorruptStorage("HNSW vector storage size overflows address space".into())
        })?;
        let mut vectors = match (&mapping, storage) {
            (Some((mapping, _)), HnswVectorStorage::ScalarInt8) => VectorArena::MappedScalarInt8 {
                bytes: Arc::clone(mapping),
                offsets: Vec::with_capacity(node_count),
                scales: Vec::with_capacity(node_count),
                squared_norms: Vec::with_capacity(node_count),
            },
            _ => VectorArena::with_capacity(storage, node_count, dimension),
        };
        let mut seen_ids = HashSet::with_capacity(node_count);
        let mut ids = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let id = decoder.string()?;
            if id.is_empty() || !seen_ids.insert(id.clone()) {
                return Err(Error::CorruptStorage(
                    "HNSW graph contains an empty or duplicate ID".into(),
                ));
            }
            let vector = match storage {
                HnswVectorStorage::F32 => {
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
                    PreparedVector::F32(vector)
                }
                HnswVectorStorage::ScalarInt8 => {
                    let scale = decoder.f32()?;
                    if !scale.is_finite() || scale <= 0.0 {
                        return Err(Error::CorruptStorage(
                            "HNSW graph contains invalid quantization parameters".into(),
                        ));
                    }
                    let code_position = decoder.position();
                    let code_bytes = decoder.take(dimension)?;
                    if let Some((_, mapping_offset)) = &mapping {
                        let squared_norm = quantized_squared_norm_bytes(code_bytes, scale);
                        vectors.push_mapped(mapping_offset + code_position, scale, squared_norm)?;
                        PreparedVector::ScalarInt8(QuantizedVector {
                            codes: Vec::new(),
                            scale,
                            squared_norm,
                        })
                    } else {
                        let codes: Vec<i8> = code_bytes
                            .iter()
                            .map(|value| i8::from_le_bytes([*value]))
                            .collect();
                        let squared_norm = quantized_squared_norm(&codes, scale);
                        PreparedVector::ScalarInt8(QuantizedVector {
                            codes,
                            scale,
                            squared_norm,
                        })
                    }
                }
            };
            if !matches!(vectors, VectorArena::MappedScalarInt8 { .. }) {
                vectors.push(vector);
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
                    layer.push(u32::try_from(decoder.usize()?).map_err(|_| {
                        Error::CorruptStorage("HNSW neighbor index exceeds u32".into())
                    })?);
                }
                neighbors.push(layer);
            }
            nodes.push(Node {
                level: (layer_count - 1) as u8,
                range_start: 0,
                neighbors,
            });
            ids.push(id.into_boxed_str());
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
            if entry >= node_count || usize::from(nodes[entry].level) < max_level {
                return Err(Error::CorruptStorage("invalid HNSW entry point".into()));
            }
        }
        for (node_index, node) in nodes.iter().enumerate() {
            for layer in &node.neighbors {
                if layer.iter().any(|neighbor| {
                    *neighbor as usize >= node_count || *neighbor as usize == node_index
                }) {
                    return Err(Error::CorruptStorage(
                        "HNSW graph contains an invalid neighbor".into(),
                    ));
                }
            }
        }

        let mut index = Self {
            dimension,
            metric,
            config,
            nodes,
            ids,
            vectors,
            entry_point,
            max_level,
            search_workspaces: Arc::new(Mutex::new(Vec::new())),
            compact_graph: None,
        };
        index.finalize_compact_graph().map_err(|error| {
            Error::CorruptStorage(format!("could not compact decoded HNSW graph: {error}"))
        })?;
        Ok(index)
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
                current = self.greedy_closest(point.vector.as_ref(), current, layer);
            }
        }

        let connected_levels = level.min(self.max_level);
        let mut neighbors = vec![Vec::new(); level + 1];
        for layer in (0..=connected_levels).rev() {
            let candidates = self.search_layer_for_build(
                point.vector.as_ref(),
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
        self.vectors.push(vector);
        self.nodes.push(Node {
            level: level as u8,
            range_start: 0,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.ids.push(id.into_boxed_str());

        if self.entry_point.is_none() {
            self.entry_point = Some(new_index);
            self.max_level = level;
            return;
        }

        for (layer, selected) in plan.neighbors.into_iter().enumerate() {
            self.nodes[new_index].neighbors[layer]
                .extend(selected.iter().map(|item| item.index as u32));
            for neighbor in selected {
                self.nodes[neighbor.index].neighbors[layer].push(new_index as u32);
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

    fn finalize_compact_graph(&mut self) -> Result<()> {
        if self.compact_graph.is_some() {
            return Ok(());
        }
        let range_count = self
            .nodes
            .iter()
            .try_fold(0_usize, |count, node| {
                count.checked_add(node.neighbors.len())
            })
            .ok_or_else(|| Error::ResourceExhausted("HNSW layer count overflows memory".into()))?;
        let link_count = self
            .nodes
            .iter()
            .try_fold(0_usize, |count, node| {
                node.neighbors
                    .iter()
                    .try_fold(count, |count, layer| count.checked_add(layer.len()))
            })
            .ok_or_else(|| Error::ResourceExhausted("HNSW link count overflows memory".into()))?;
        if range_count > u32::MAX as usize || link_count > u32::MAX as usize {
            return Err(Error::ResourceExhausted(
                "HNSW compact graph exceeds u32 address space".into(),
            ));
        }

        let mut ranges = Vec::with_capacity(range_count);
        let mut links = Vec::with_capacity(link_count);
        for node in &mut self.nodes {
            node.range_start = ranges.len() as u32;
            for layer in node.neighbors.drain(..) {
                let start = links.len() as u32;
                links.extend(layer);
                ranges.push(LinkRange {
                    start,
                    end: links.len() as u32,
                });
            }
            node.neighbors.shrink_to_fit();
        }
        self.compact_graph = Some(CompactGraph { ranges, links });
        Ok(())
    }

    fn prune(&mut self, node: usize, layer: usize) {
        let limit = max_connections(self.config, layer);
        if self.nodes[node].neighbors[layer].len() <= limit {
            return;
        }
        let mut candidates: Vec<HeapItem> = self.nodes[node].neighbors[layer]
            .iter()
            .map(|&index| HeapItem {
                index: index as usize,
                score: score_vectors(self.metric, self.vector(node), self.vector(index as usize)),
            })
            .collect();
        self.sort_items(&mut candidates);
        self.nodes[node].neighbors[layer] = self
            .select_neighbors(candidates, limit)
            .into_iter()
            .map(|item| item.index as u32)
            .collect();
    }

    fn greedy_closest(&self, query: VectorRef<'_>, start: usize, layer: usize) -> usize {
        let mut current = start;
        loop {
            let mut best = current;
            let mut best_score = self.score(query, current);
            for &neighbor in self.neighbors(current, layer) {
                let neighbor = neighbor as usize;
                let score = self.score(query, neighbor);
                if compare_scored(score, &self.ids[neighbor], best_score, &self.ids[best])
                    == Ordering::Less
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

    fn search_layer(
        &self,
        query: VectorRef<'_>,
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
                let neighbor = neighbor as usize;
                if !workspace.visit(neighbor) {
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

    fn take_search_workspace(&self) -> BuildWorkspace {
        self.search_workspaces
            .lock()
            .ok()
            .and_then(|mut workspaces| workspaces.pop())
            .unwrap_or_else(|| BuildWorkspace::with_capacity(self.nodes.len()))
    }

    fn return_search_workspace(&self, workspace: BuildWorkspace) {
        if let Ok(mut workspaces) = self.search_workspaces.lock()
            && workspaces.len() < MAX_POOLED_SEARCH_WORKSPACES
        {
            workspaces.push(workspace);
        }
    }

    fn search_layer_for_build(
        &self,
        query: VectorRef<'_>,
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
                let neighbor = neighbor as usize;
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
                score_vectors(
                    self.metric,
                    self.vector(candidate.index),
                    self.vector(other.index),
                ) < candidate.score
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
                &self.ids[left.index],
                right.score,
                &self.ids[right.index],
            )
        });
    }

    fn neighbors(&self, node: usize, layer: usize) -> &[u32] {
        let node = &self.nodes[node];
        if layer > usize::from(node.level) {
            return &[];
        }
        if let Some(graph) = &self.compact_graph {
            let range = graph.ranges[node.range_start as usize + layer];
            &graph.links[range.start as usize..range.end as usize]
        } else {
            node.neighbors
                .get(layer)
                .map(Vec::as_slice)
                .unwrap_or_default()
        }
    }

    fn score(&self, query: VectorRef<'_>, node: usize) -> f32 {
        score_vectors(self.metric, query, self.vector(node))
    }

    fn vector(&self, node: usize) -> VectorRef<'_> {
        self.vectors.get(node, self.dimension)
    }
}

fn score_vectors(metric: Metric, left: VectorRef<'_>, right: VectorRef<'_>) -> f32 {
    match (left, right) {
        (VectorRef::F32(left), VectorRef::F32(right)) => metric.score(left, right),
        (
            VectorRef::ScalarInt8 {
                codes: left,
                scale: left_scale,
                squared_norm: left_norm,
            },
            VectorRef::ScalarInt8 {
                codes: right,
                scale: right_scale,
                squared_norm: right_norm,
            },
        ) => {
            let dot = quantized_dot(left, left_scale, right, right_scale);
            match metric {
                Metric::Cosine => {
                    let denominator = (left_norm * right_norm).sqrt();
                    if denominator > 0.0 {
                        dot / denominator
                    } else {
                        0.0
                    }
                }
                Metric::DotProduct => dot,
                Metric::Euclidean => -(left_norm + right_norm - 2.0 * dot).max(0.0),
            }
        }
        _ => unreachable!("HNSW queries and stored vectors use the same representation"),
    }
}

fn quantized_dot(left: &[i8], left_scale: f32, right: &[i8], right_scale: f32) -> f32 {
    quantized_integer_dot(left, right) as f32 * left_scale * right_scale
}

fn quantized_integer_dot(left: &[i8], right: &[i8]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "x86_64")]
    if left.len() <= 65_536 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability is checked at runtime and both slices have
        // the same length. The implementation uses unaligned, in-bounds loads.
        return unsafe { quantized_integer_dot_avx2(left, right) };
    }
    quantized_integer_dot_portable(left, right)
}

fn quantized_integer_dot_portable(left: &[i8], right: &[i8]) -> i64 {
    let mut sums = [0_i64; 4];
    let mut left_chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);
    for (left, right) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        sums[0] += i64::from(left[0]) * i64::from(right[0]);
        sums[1] += i64::from(left[1]) * i64::from(right[1]);
        sums[2] += i64::from(left[2]) * i64::from(right[2]);
        sums[3] += i64::from(left[3]) * i64::from(right[3]);
    }
    let tail = left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
        .map(|(left, right)| i64::from(*left) * i64::from(*right))
        .sum::<i64>();
    sums.into_iter().sum::<i64>() + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantized_integer_dot_avx2(left: &[i8], right: &[i8]) -> i64 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi32, _mm256_cvtepi8_epi16,
        _mm256_madd_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
    };

    let vectorized = left.len() / 16 * 16;
    let mut sums = _mm256_setzero_si256();
    let mut offset = 0;
    while offset < vectorized {
        // SAFETY: offset advances in 16-byte chunks and remains below the
        // rounded-down vectorized length. Unaligned loads are intentional.
        let left_bytes = unsafe { _mm_loadu_si128(left.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: same bounds argument as the left-hand load.
        let right_bytes = unsafe { _mm_loadu_si128(right.as_ptr().add(offset).cast::<__m128i>()) };
        let left_words = _mm256_cvtepi8_epi16(left_bytes);
        let right_words = _mm256_cvtepi8_epi16(right_bytes);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(left_words, right_words));
        offset += 16;
    }

    let mut lanes = [0_i32; 8];
    // SAFETY: lanes has exactly the 32 bytes required by the unaligned store.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums) };
    lanes.into_iter().map(i64::from).sum::<i64>()
        + quantized_integer_dot_portable(&left[vectorized..], &right[vectorized..])
}

fn quantized_squared_norm(codes: &[i8], scale: f32) -> f32 {
    quantized_dot(codes, scale, codes, scale)
}

fn quantized_squared_norm_bytes(codes: &[u8], scale: f32) -> f32 {
    let sum = codes.iter().fold(0_i64, |sum, &code| {
        let code = i64::from(i8::from_le_bytes([code]));
        sum + code * code
    });
    sum as f32 * scale * scale
}

#[derive(Debug)]
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

fn storage_tag(storage: HnswVectorStorage) -> u8 {
    match storage {
        HnswVectorStorage::F32 => 1,
        HnswVectorStorage::ScalarInt8 => 2,
    }
}

fn parse_storage(tag: u8) -> Result<HnswVectorStorage> {
    match tag {
        1 => Ok(HnswVectorStorage::F32),
        2 => Ok(HnswVectorStorage::ScalarInt8),
        _ => Err(Error::CorruptStorage(format!(
            "unknown HNSW vector storage tag {tag}"
        ))),
    }
}

fn put_u32_writer(output: &mut impl Write, value: usize) -> Result<()> {
    let value = u32::try_from(value)
        .map_err(|_| Error::InvalidQuery("HNSW graph value exceeds format limit"))?;
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn put_bytes_writer(output: &mut impl Write, value: &[u8]) -> Result<()> {
    put_u32_writer(output, value.len())?;
    output.write_all(value)?;
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

    fn position(&self) -> usize {
        self.position
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
    use std::thread;

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
    fn pooled_search_workspaces_are_safe_under_contention() {
        let index = Arc::new(
            HnswIndex::build_quantized(3, Metric::Cosine, HnswConfig::default(), points(1_000))
                .unwrap(),
        );
        let workers: Vec<_> = (0..16)
            .map(|worker| {
                let index = Arc::clone(&index);
                thread::spawn(move || {
                    for query in 0..100 {
                        let angle = (worker * 100 + query) as f32 * 0.137;
                        let hits = index
                            .search(vec![angle.cos(), angle.sin(), (angle * 0.31).cos()], 10, 64)
                            .unwrap();
                        assert_eq!(hits.len(), 10);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(index.search_workspaces.lock().unwrap().len() <= MAX_POOLED_SEARCH_WORKSPACES);
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
        let encoded = original.encode().unwrap();
        assert_eq!(encoded.len(), original.encoded_len().unwrap());
        let restored = HnswIndex::decode(&encoded).unwrap();
        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.config(), original.config());
        assert_eq!(
            restored.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap(),
            original.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap()
        );
    }

    #[test]
    fn quantized_search_supports_every_metric_and_zero_vectors() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let mut input = vec![
                ("a".into(), vec![10.0, 0.2, -0.1, 0.0]),
                ("b".into(), vec![0.1, 9.0, 0.2, 0.0]),
                ("c".into(), vec![-8.0, 0.1, 0.0, 0.0]),
            ];
            if metric != Metric::Cosine {
                input.push(("zero".into(), vec![0.0; 4]));
            }
            let index = HnswIndex::build_quantized(
                4,
                metric,
                HnswConfig {
                    m: 8,
                    ef_construction: 32,
                },
                input,
            )
            .unwrap();
            let hits = index.search(vec![9.0, 0.1, 0.0, 0.0], 2, 8).unwrap();
            assert_eq!(hits[0].id, "a", "metric {metric:?}");
            assert!(hits.iter().all(|hit| hit.score.is_finite()));
        }
    }

    #[test]
    fn quantized_vectors_use_about_one_quarter_of_f32_storage() {
        let input: Vec<_> = (0..100)
            .map(|index| {
                (
                    format!("p-{index}"),
                    (0..768)
                        .map(|dimension| ((index * 769 + dimension) as f32 * 0.017).sin())
                        .collect(),
                )
            })
            .collect();
        let config = HnswConfig {
            m: 8,
            ef_construction: 32,
        };
        let full = HnswIndex::build(768, Metric::Cosine, config, input.clone()).unwrap();
        let quantized = HnswIndex::build_quantized(768, Metric::Cosine, config, input).unwrap();
        assert_eq!(full.vector_storage_bytes(), 100 * 768 * 4);
        assert_eq!(quantized.vector_storage_bytes(), 100 * (768 + 8));
        assert!(quantized.vector_storage_bytes() * 3 < full.vector_storage_bytes());
        assert!(quantized.encode().unwrap().len() * 2 < full.encode().unwrap().len());
    }

    #[test]
    fn quantized_graph_round_trips_for_every_metric() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let original = HnswIndex::build_quantized(
                3,
                metric,
                HnswConfig {
                    m: 8,
                    ef_construction: 40,
                },
                points(100),
            )
            .unwrap();
            let encoded = original.encode().unwrap();
            let restored = HnswIndex::decode(&encoded).unwrap();
            assert_eq!(restored.vector_storage(), HnswVectorStorage::ScalarInt8);
            assert_eq!(
                restored.vector_storage_bytes(),
                original.vector_storage_bytes()
            );
            assert_eq!(
                restored.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap(),
                original.search(vec![0.2, 0.8, 0.4], 10, 32).unwrap(),
                "metric {metric:?}"
            );
        }
    }

    #[test]
    fn decoder_accepts_version_one_f32_graphs() {
        let index = HnswIndex::build(
            3,
            Metric::Cosine,
            HnswConfig {
                m: 8,
                ef_construction: 32,
            },
            points(20),
        )
        .unwrap();
        let mut legacy = index.encode().unwrap();
        assert_eq!(legacy[4], GRAPH_VERSION);
        assert_eq!(legacy[14], storage_tag(HnswVectorStorage::F32));
        legacy[4] = LEGACY_GRAPH_VERSION;
        legacy.remove(14);
        let restored = HnswIndex::decode(&legacy).unwrap();
        assert_eq!(restored.vector_storage(), HnswVectorStorage::F32);
        assert_eq!(
            restored.search(vec![0.2, 0.8, 0.4], 5, 16).unwrap(),
            index.search(vec![0.2, 0.8, 0.4], 5, 16).unwrap()
        );
    }

    #[test]
    fn decoder_rejects_corrupt_quantization_parameters() {
        let index = HnswIndex::build_quantized(
            2,
            Metric::DotProduct,
            HnswConfig::default(),
            [("a".into(), vec![1.0, 2.0])],
        )
        .unwrap();
        let mut encoded = index.encode().unwrap();
        // Fixed header (39 bytes), then the four-byte ID length and one-byte ID.
        encoded[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            HnswIndex::decode(&encoded),
            Err(Error::CorruptStorage(_))
        ));
    }

    #[test]
    fn quantized_parallel_builds_are_deterministic() {
        let input = points(PARALLEL_BUILD_SEED + 128);
        let build = || {
            HnswIndex::build_quantized(
                3,
                Metric::Cosine,
                HnswConfig {
                    m: 8,
                    ef_construction: 32,
                },
                input.clone(),
            )
            .unwrap()
        };
        assert_eq!(build().encode().unwrap(), build().encode().unwrap());
    }

    #[test]
    fn graph_is_identical_across_rayon_pool_sizes() {
        let input = points(PARALLEL_BUILD_SEED + 128);
        for storage in [HnswVectorStorage::F32, HnswVectorStorage::ScalarInt8] {
            let build = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        HnswIndex::build_with_storage(
                            3,
                            Metric::Cosine,
                            HnswConfig {
                                m: 8,
                                ef_construction: 32,
                            },
                            storage,
                            input.clone(),
                        )
                        .unwrap()
                        .encode()
                        .unwrap()
                    })
            };
            assert_eq!(build(1), build(4), "storage {storage:?}");
        }
    }

    #[test]
    fn quantized_candidate_recall_is_high() {
        let input: Vec<_> = (0..1_000)
            .map(|index| {
                let vector = (0..64)
                    .map(|dimension| {
                        ((index * 67 + dimension * 13) as f32 * 0.019).sin()
                            + ((index + dimension * 7) as f32 * 0.007).cos()
                    })
                    .collect();
                (format!("point-{index:04}"), vector)
            })
            .collect();
        let index = HnswIndex::build_quantized(
            64,
            Metric::Cosine,
            HnswConfig {
                m: 16,
                ef_construction: 120,
            },
            input.clone(),
        )
        .unwrap();
        let mut matches = 0;
        let mut total = 0;
        for query_index in (0..1_000).step_by(29) {
            let query = input[query_index].1.clone();
            let candidates: HashSet<_> = index
                .search(query.clone(), 40, 128)
                .unwrap()
                .into_iter()
                .map(|hit| hit.id)
                .collect();
            let prepared = Metric::Cosine.prepare(query).unwrap();
            let mut exact = input.clone();
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
                matches += usize::from(candidates.contains(id));
            }
        }
        let recall = matches as f64 / total as f64;
        assert!(
            recall >= 0.98,
            "quantized candidate recall@10 was {recall:.3}"
        );
    }

    #[test]
    fn scalar_int8_scores_track_f32_across_metrics() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            for sample in 0..500 {
                let left: Vec<_> = (0..64)
                    .map(|dimension| ((sample * 67 + dimension * 19) as f32 * 0.013).sin() * 3.7)
                    .collect();
                let right: Vec<_> = (0..64)
                    .map(|dimension| ((sample * 29 + dimension * 43) as f32 * 0.011).cos() * 2.3)
                    .collect();
                let left = metric.prepare(left).unwrap();
                let right = metric.prepare(right).unwrap();
                let exact = metric.score(&left, &right);
                let quantized = score_vectors(
                    metric,
                    QuantizedVector::new(&left).as_ref(),
                    QuantizedVector::new(&right).as_ref(),
                );
                let scale = match metric {
                    Metric::Cosine => 1.0,
                    Metric::DotProduct => {
                        let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
                        let right_norm =
                            right.iter().map(|value| value * value).sum::<f32>().sqrt();
                        1.0 + left_norm * right_norm
                    }
                    Metric::Euclidean => 1.0 + exact.abs(),
                };
                let relative_error = (exact - quantized).abs() / scale;
                assert!(
                    relative_error < 0.025,
                    "metric {metric:?} sample {sample}: exact={exact} quantized={quantized} error={relative_error}"
                );
            }
        }
    }

    #[test]
    fn simd_integer_dot_matches_portable_at_boundaries_and_extremes() {
        for length in [0, 1, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129, 4_096] {
            let left: Vec<_> = (0..length)
                .map(|index| match index % 5 {
                    0 => -127,
                    1 => 127,
                    2 => -1,
                    3 => 1,
                    _ => ((index * 37) % 255) as i16 as i8,
                })
                .collect();
            let right: Vec<_> = (0..length)
                .map(|index| match index % 7 {
                    0 => 127,
                    1 => -127,
                    2 => 1,
                    3 => -1,
                    _ => ((index * 53) % 255) as i16 as i8,
                })
                .collect();
            let expected = quantized_integer_dot_portable(&left, &right);
            assert_eq!(quantized_integer_dot(&left, &right), expected);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime AVX2 detection.
                assert_eq!(
                    unsafe { quantized_integer_dot_avx2(&left, &right) },
                    expected
                );
            }
        }
    }
}
