use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::MmapOptions;

use crate::metadata_index::MetadataIndex;
use crate::{
    Collection, CollectionConfig, Error, Filter, HnswConfig, HnswIndex, Metric, Point, Result,
    SearchHit, UpsertPoint, Value,
};

const META_MAGIC: &[u8; 4] = b"FVMT";
const WAL_MAGIC: &[u8; 4] = b"FVWL";
const SEGMENT_MAGIC: &[u8; 4] = b"FVSG";
const MANIFEST_MAGIC: &[u8; 4] = b"FVMF";
const FORMAT_VERSION: u8 = 1;
const SEGMENT_VERSION: u8 = 2;
const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;
const MAX_ITEMS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Flush bytes to the operating system and stable storage before returning.
    Sync,
    /// Flush bytes to the operating system before returning. A power loss can
    /// lose acknowledged writes, but process crashes remain recoverable.
    Flush,
}

#[derive(Debug)]
pub struct PersistentCollection {
    _lock: File,
    inner: Collection,
    wal: File,
    durability: Durability,
    directory: PathBuf,
    index: Option<HnswIndex>,
    metadata_index: Option<MetadataIndex>,
    dirty_ids: HashSet<String>,
}

pub(crate) struct FlushSnapshot {
    config: CollectionConfig,
    sequence: u64,
    points: Vec<Point>,
}

pub(crate) struct PreparedFlush {
    sequence: u64,
    bytes: Vec<u8>,
    index: HnswIndex,
    metadata_index: MetadataIndex,
}

impl PersistentCollection {
    pub fn open(
        directory: impl AsRef<Path>,
        config: CollectionConfig,
        durability: Durability,
    ) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("collection.lock"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(Error::Concurrency(
                    "collection directory is already open by another process or handle".into(),
                ));
            }
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }
        ensure_metadata(&directory, config)?;

        let wal_path = directory.join("write.wal");
        let mut wal = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(wal_path)?;
        let mut inner = Collection::new(config)?;
        let loaded = load_current_segment(&directory, &mut inner)?;
        let (index, metadata_index) = loaded
            .map(|(index, metadata)| (Some(index), Some(metadata)))
            .unwrap_or((None, None));
        let mut dirty_ids = HashSet::new();
        recover(&mut wal, &mut inner, &mut dirty_ids)?;
        wal.seek(SeekFrom::End(0))?;

        Ok(Self {
            _lock: lock,
            inner,
            wal,
            durability,
            directory,
            index,
            metadata_index,
            dirty_ids,
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn config(&self) -> CollectionConfig {
        self.inner.config()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn persisted_config(directory: impl AsRef<Path>) -> Result<CollectionConfig> {
        let mut bytes = Vec::new();
        File::open(directory.as_ref().join("collection.meta"))?.read_to_end(&mut bytes)?;
        decode_metadata(&bytes)
    }

    pub fn collection(&self) -> &Collection {
        &self.inner
    }

    pub fn latest_sequence(&self) -> u64 {
        self.inner.latest_sequence()
    }

    pub fn upsert(&mut self, points: Vec<UpsertPoint>) -> Result<u64> {
        let sequence = self
            .inner
            .latest_sequence()
            .checked_add(1)
            .ok_or(Error::SequenceOverflow)?;
        let prepared = self.inner.prepare_upsert(points, sequence)?;
        let changed_ids: Vec<String> = prepared.iter().map(|point| point.id.clone()).collect();
        let payload = encode_upsert(sequence, &prepared)?;
        self.append_frame(&payload)?;
        self.inner.apply_prepared_upsert(prepared, sequence);
        self.dirty_ids.extend(changed_ids);
        Ok(sequence)
    }

    pub fn delete(&mut self, ids: Vec<String>) -> Result<u64> {
        if ids.is_empty() {
            return Err(Error::InvalidQuery("delete batch must not be empty"));
        }
        if ids.iter().any(String::is_empty) {
            return Err(Error::InvalidQuery("point ID must not be empty"));
        }
        let sequence = self
            .inner
            .latest_sequence()
            .checked_add(1)
            .ok_or(Error::SequenceOverflow)?;
        let payload = encode_delete(sequence, &ids)?;
        self.append_frame(&payload)?;
        self.inner.apply_delete_at(&ids, sequence);
        self.dirty_ids.extend(ids);
        Ok(sequence)
    }

    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchHit>> {
        self.search_with_ef(query, k, filter, k.saturating_mul(4).max(96))
    }

    pub fn search_with_ef(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
        ef_search: usize,
    ) -> Result<Vec<SearchHit>> {
        if ef_search < k {
            return Err(Error::InvalidQuery("ef_search must be at least k"));
        }
        if let Some(filter) = filter {
            let Some(metadata_index) = &self.metadata_index else {
                return self.inner.search(query, k, Some(filter));
            };
            let mut candidates = metadata_index.candidates(filter);
            candidates.retain(|id| !self.dirty_ids.contains(id));
            let mut hits = self
                .inner
                .search_ids(query.clone(), k, &candidates, Some(filter))?;
            hits.extend(
                self.inner
                    .search_ids(query, k, &self.dirty_ids, Some(filter))?,
            );
            return Ok(merge_hits(hits, k));
        }

        let Some(index) = self.index.as_ref() else {
            return self.inner.search(query, k, None);
        };
        let graph_k = k.saturating_add(self.dirty_ids.len()).min(index.len());
        let mut hits = Vec::with_capacity(k.saturating_add(self.dirty_ids.len()));
        if graph_k > 0 {
            for hit in index.search(query.clone(), graph_k, ef_search.max(graph_k))? {
                if self.dirty_ids.contains(&hit.id) {
                    continue;
                }
                let point = self.inner.get(&hit.id).ok_or_else(|| {
                    Error::CorruptStorage(format!(
                        "HNSW point {} is missing from its segment",
                        hit.id
                    ))
                })?;
                hits.push(SearchHit {
                    id: hit.id,
                    score: hit.score,
                    metadata: point.metadata.clone(),
                    sequence: point.sequence,
                });
            }
        }
        hits.extend(self.inner.search_ids(query, k, &self.dirty_ids, None)?);
        Ok(merge_hits(hits, k))
    }

    pub fn has_approximate_index(&self) -> bool {
        self.index.is_some()
    }

    pub fn pending_point_count(&self) -> usize {
        self.dirty_ids.len()
    }

    pub(crate) fn copy_backup_to(&self, destination: &Path) -> Result<()> {
        copy_collection_directory(&self.directory, destination)
    }

    pub(crate) fn restore_backup(source: &Path, destination: &Path) -> Result<CollectionConfig> {
        let config = Self::persisted_config(source)?;
        copy_collection_directory(source, destination)?;
        Ok(config)
    }

    /// Writes a complete immutable snapshot and checkpoints the WAL.
    ///
    /// Publication order is segment, manifest, then WAL truncation, with each
    /// durable before the next. A crash at any boundary therefore recovers
    /// from either the old log or the newly published segment.
    pub fn flush(&mut self) -> Result<u64> {
        let snapshot = self.flush_snapshot();
        let prepared = Self::build_flush(snapshot)?;
        self.publish_flush(prepared)
    }

    pub(crate) fn flush_snapshot(&self) -> FlushSnapshot {
        FlushSnapshot {
            config: self.inner.config(),
            sequence: self.inner.latest_sequence(),
            points: self.inner.snapshot_points(),
        }
    }

    pub(crate) fn build_flush(snapshot: FlushSnapshot) -> Result<PreparedFlush> {
        let metadata_index = MetadataIndex::build(&snapshot.points);
        let index = HnswIndex::build(
            snapshot.config.dimension,
            snapshot.config.metric,
            HnswConfig::default(),
            snapshot
                .points
                .iter()
                .map(|point| (point.id.clone(), point.vector.clone())),
        )?;
        let bytes = encode_segment(snapshot.config, snapshot.sequence, &snapshot.points, &index)?;
        Ok(PreparedFlush {
            sequence: snapshot.sequence,
            bytes,
            index,
            metadata_index,
        })
    }

    pub(crate) fn publish_flush(&mut self, prepared: PreparedFlush) -> Result<u64> {
        let previous = read_manifest(&self.directory)?;
        if let Some((published_sequence, _)) = &previous
            && *published_sequence > prepared.sequence
        {
            return Ok(*published_sequence);
        }

        let sequence = prepared.sequence;
        let segment_name = format!("segment-{sequence:020}.fvs");
        let segment_path = self.directory.join(&segment_name);
        let temporary_segment = self
            .directory
            .join(format!(".{segment_name}.tmp-{}", std::process::id()));

        write_new_file(&temporary_segment, &prepared.bytes)?;
        fs::rename(&temporary_segment, &segment_path)?;
        sync_directory(&self.directory)?;

        publish_manifest(&self.directory, sequence, &segment_name)?;

        if self.inner.latest_sequence() == sequence {
            self.wal.set_len(0)?;
            self.wal.seek(SeekFrom::Start(0))?;
            self.wal.sync_data()?;
            self.dirty_ids.clear();
        }

        if let Some((_, previous_name)) = previous
            && previous_name != segment_name
        {
            let _ = fs::remove_file(self.directory.join(previous_name));
        }
        self.index = Some(prepared.index);
        self.metadata_index = Some(prepared.metadata_index);
        Ok(sequence)
    }

    fn append_frame(&mut self, payload: &[u8]) -> Result<()> {
        let length = u32::try_from(payload.len())
            .map_err(|_| Error::InvalidQuery("WAL record is too large"))?;
        let frame_start = self.wal.seek(SeekFrom::End(0))?;
        let result = (|| -> std::io::Result<()> {
            self.wal.write_all(WAL_MAGIC)?;
            self.wal.write_all(&length.to_le_bytes())?;
            self.wal.write_all(payload)?;
            self.wal.write_all(&crc32c(payload).to_le_bytes())?;
            match self.durability {
                Durability::Sync => self.wal.sync_data()?,
                Durability::Flush => self.wal.flush()?,
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Keep this process usable after a short or failed append. Recovery
            // still independently handles a process dying before this cleanup.
            let _ = self.wal.set_len(frame_start);
            let _ = self.wal.seek(SeekFrom::End(0));
            return Err(error.into());
        }
        Ok(())
    }
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)?.sync_all()?;
    Ok(())
}

fn copy_collection_directory(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        return Err(Error::AlreadyExists(format!(
            "backup destination {}",
            destination.display()
        )));
    }
    let parent = destination
        .parent()
        .ok_or(Error::InvalidQuery("backup destination has no parent"))?;
    fs::create_dir_all(parent)?;
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(Error::InvalidQuery("backup destination name is invalid"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::Io(error.to_string()))?
        .as_nanos();
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&temporary)?;

    let result = (|| -> Result<()> {
        copy_and_sync(
            &source.join("collection.meta"),
            &temporary.join("collection.meta"),
        )?;
        copy_and_sync(&source.join("write.wal"), &temporary.join("write.wal"))?;
        if let Some((_, segment_name)) = read_manifest(source)? {
            copy_and_sync(&source.join(&segment_name), &temporary.join(&segment_name))?;
            copy_and_sync(&source.join("MANIFEST"), &temporary.join("MANIFEST"))?;
        }
        sync_directory(&temporary)?;
        fs::rename(&temporary, destination)?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn copy_and_sync(source: &Path, destination: &Path) -> Result<()> {
    fs::copy(source, destination)?;
    File::open(destination)?.sync_all()?;
    Ok(())
}

fn publish_manifest(directory: &Path, sequence: u64, segment_name: &str) -> Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MANIFEST_MAGIC);
    bytes.push(FORMAT_VERSION);
    bytes.extend_from_slice(&sequence.to_le_bytes());
    put_string(&mut bytes, segment_name)?;
    let checksum = crc32c(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());

    let temporary = directory.join(format!(".MANIFEST.tmp-{}", std::process::id()));
    write_new_file(&temporary, &bytes)?;
    fs::rename(temporary, directory.join("MANIFEST"))?;
    sync_directory(directory)
}

fn read_manifest(directory: &Path) -> Result<Option<(u64, String)>> {
    let path = directory.join("MANIFEST");
    if !path.exists() {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 21 || &bytes[..4] != MANIFEST_MAGIC || bytes[4] != FORMAT_VERSION {
        return Err(Error::CorruptStorage("invalid manifest header".into()));
    }
    let payload_length = bytes.len() - 4;
    let stored_crc = u32::from_le_bytes(
        bytes[payload_length..]
            .try_into()
            .expect("four-byte checksum"),
    );
    if crc32c(&bytes[..payload_length]) != stored_crc {
        return Err(Error::CorruptStorage("manifest checksum mismatch".into()));
    }
    let mut decoder = Decoder::new(&bytes[5..payload_length]);
    let sequence = decoder.u64()?;
    let segment_name = decoder.string()?;
    decoder.finish()?;
    if segment_name.contains('/') || !segment_name.starts_with("segment-") {
        return Err(Error::CorruptStorage(
            "invalid segment name in manifest".into(),
        ));
    }
    Ok(Some((sequence, segment_name)))
}

fn load_current_segment(
    directory: &Path,
    collection: &mut Collection,
) -> Result<Option<(HnswIndex, MetadataIndex)>> {
    let Some((manifest_sequence, segment_name)) = read_manifest(directory)? else {
        return Ok(None);
    };
    let segment = File::open(directory.join(segment_name))?;
    if segment.metadata()?.len() == 0 {
        return Err(Error::CorruptStorage("segment file is empty".into()));
    }
    // SAFETY: Published segment files are immutable. This engine replaces
    // manifests and segment pathnames atomically and never mutates a published
    // segment inode. The map lives only for decoding and is dropped before any
    // obsolete segment can be retired by this collection instance.
    let bytes = unsafe { MmapOptions::new().map(&segment)? };
    let (config, sequence, points, index) = decode_segment(&bytes)?;
    if config != collection.config() {
        return Err(Error::CorruptStorage(
            "segment configuration differs from collection metadata".into(),
        ));
    }
    if sequence != manifest_sequence {
        return Err(Error::CorruptStorage(
            "manifest and segment sequences differ".into(),
        ));
    }
    let metadata_index = MetadataIndex::build(&points);
    collection.restore_snapshot(points, sequence)?;
    Ok(Some((index, metadata_index)))
}

fn merge_hits(mut hits: Vec<SearchHit>, k: usize) -> Vec<SearchHit> {
    hits.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits.dedup_by(|left, right| left.id == right.id);
    hits.truncate(k);
    hits
}

fn encode_segment(
    config: CollectionConfig,
    sequence: u64,
    points: &[Point],
    index: &HnswIndex,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.extend_from_slice(SEGMENT_MAGIC);
    output.push(SEGMENT_VERSION);
    output.push(metric_tag(config.metric));
    output.extend_from_slice(&(config.dimension as u64).to_le_bytes());
    output.extend_from_slice(&sequence.to_le_bytes());
    put_count(&mut output, points.len())?;
    for point in points {
        put_string(&mut output, &point.id)?;
        output.extend_from_slice(&point.sequence.to_le_bytes());
        for value in &point.vector {
            output.extend_from_slice(&value.to_le_bytes());
        }
        put_count(&mut output, point.metadata.len())?;
        for (key, value) in &point.metadata {
            put_string(&mut output, key)?;
            encode_value(&mut output, value)?;
        }
    }
    let graph = index.encode()?;
    put_count(&mut output, graph.len())?;
    output.extend_from_slice(&graph);
    let checksum = crc32c(&output);
    output.extend_from_slice(&checksum.to_le_bytes());
    Ok(output)
}

fn decode_segment(bytes: &[u8]) -> Result<(CollectionConfig, u64, Vec<Point>, HnswIndex)> {
    if bytes.len() < 30 || &bytes[..4] != SEGMENT_MAGIC || bytes[4] != SEGMENT_VERSION {
        return Err(Error::CorruptStorage("invalid segment header".into()));
    }
    let payload_length = bytes.len() - 4;
    let stored_crc = u32::from_le_bytes(
        bytes[payload_length..]
            .try_into()
            .expect("four-byte checksum"),
    );
    if crc32c(&bytes[..payload_length]) != stored_crc {
        return Err(Error::CorruptStorage("segment checksum mismatch".into()));
    }
    let metric = parse_metric(bytes[5])?;
    let dimension_u64 = u64::from_le_bytes(bytes[6..14].try_into().expect("fixed slice"));
    let dimension = usize::try_from(dimension_u64)
        .map_err(|_| Error::CorruptStorage("segment dimension is too large".into()))?;
    if dimension == 0 || dimension > MAX_ITEMS {
        return Err(Error::CorruptStorage("invalid segment dimension".into()));
    }
    let mut decoder = Decoder::new(&bytes[14..payload_length]);
    let sequence = decoder.u64()?;
    let count = decoder.count()?;
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        let id = decoder.string()?;
        let point_sequence = decoder.u64()?;
        let mut vector = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            vector.push(decoder.f32()?);
        }
        let metadata_count = decoder.count()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            metadata.insert(decoder.string()?, decoder.value()?);
        }
        points.push(Point {
            id,
            vector,
            metadata,
            sequence: point_sequence,
        });
    }
    let graph_length = decoder.u32()? as usize;
    let index = HnswIndex::decode(decoder.take(graph_length)?)?;
    decoder.finish()?;
    if index.dimension() != dimension || index.metric() != metric || index.len() != points.len() {
        return Err(Error::CorruptStorage(
            "HNSW graph does not match its segment".into(),
        ));
    }
    let point_ids: std::collections::BTreeSet<&str> =
        points.iter().map(|point| point.id.as_str()).collect();
    let index_ids: std::collections::BTreeSet<&str> = index.ids().collect();
    if point_ids != index_ids {
        return Err(Error::CorruptStorage(
            "HNSW graph IDs do not match its segment".into(),
        ));
    }
    Ok((
        CollectionConfig { dimension, metric },
        sequence,
        points,
        index,
    ))
}

fn ensure_metadata(directory: &Path, config: CollectionConfig) -> Result<()> {
    let path = directory.join("collection.meta");
    if path.exists() {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let actual = decode_metadata(&bytes)?;
        if actual != config {
            return Err(Error::InvalidConfig(
                "requested configuration differs from persisted collection",
            ));
        }
        return Ok(());
    }

    let bytes = encode_metadata(config);
    let temporary = directory.join("collection.meta.tmp");
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    File::open(directory)?.sync_all()?;
    Ok(())
}

fn encode_metadata(config: CollectionConfig) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18);
    bytes.extend_from_slice(META_MAGIC);
    bytes.push(FORMAT_VERSION);
    bytes.push(metric_tag(config.metric));
    bytes.extend_from_slice(&(config.dimension as u64).to_le_bytes());
    let checksum = crc32c(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

fn decode_metadata(bytes: &[u8]) -> Result<CollectionConfig> {
    if bytes.len() != 18 || &bytes[..4] != META_MAGIC || bytes[4] != FORMAT_VERSION {
        return Err(Error::CorruptStorage(
            "invalid collection metadata header".into(),
        ));
    }
    let stored_crc = u32::from_le_bytes(bytes[14..18].try_into().expect("fixed slice"));
    if crc32c(&bytes[..14]) != stored_crc {
        return Err(Error::CorruptStorage(
            "collection metadata checksum mismatch".into(),
        ));
    }
    let metric = parse_metric(bytes[5])?;
    let dimension_u64 = u64::from_le_bytes(bytes[6..14].try_into().expect("fixed slice"));
    let dimension = usize::try_from(dimension_u64)
        .map_err(|_| Error::CorruptStorage("collection dimension is too large".into()))?;
    if dimension == 0 {
        return Err(Error::CorruptStorage(
            "collection dimension must be non-zero".into(),
        ));
    }
    Ok(CollectionConfig { dimension, metric })
}

fn recover(
    wal: &mut File,
    collection: &mut Collection,
    dirty_ids: &mut HashSet<String>,
) -> Result<()> {
    wal.seek(SeekFrom::Start(0))?;
    let mut valid_length = 0_u64;
    loop {
        let frame_start = wal.stream_position()?;
        let mut header = [0_u8; 8];
        match read_exact_or_eof(wal, &mut header)? {
            ReadState::Eof => break,
            ReadState::Truncated => {
                wal.set_len(valid_length)?;
                break;
            }
            ReadState::Complete => {}
        }
        if &header[..4] != WAL_MAGIC {
            return Err(Error::CorruptStorage(format!(
                "invalid WAL magic at byte {frame_start}"
            )));
        }
        let length = u32::from_le_bytes(header[4..8].try_into().expect("fixed slice")) as usize;
        if length > MAX_FRAME_BYTES {
            return Err(Error::CorruptStorage(format!(
                "WAL frame at byte {frame_start} exceeds the size limit"
            )));
        }
        let mut frame = vec![0_u8; length + 4];
        if read_exact_or_eof(wal, &mut frame)? != ReadState::Complete {
            wal.set_len(valid_length)?;
            break;
        }
        let stored_crc = u32::from_le_bytes(frame[length..].try_into().expect("fixed slice"));
        if crc32c(&frame[..length]) != stored_crc {
            return Err(Error::CorruptStorage(format!(
                "WAL checksum mismatch at byte {frame_start}"
            )));
        }
        dirty_ids.extend(apply_payload(collection, &frame[..length])?);
        valid_length = wal.stream_position()?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadState {
    Complete,
    Eof,
    Truncated,
}

fn read_exact_or_eof(reader: &mut File, buffer: &mut [u8]) -> Result<ReadState> {
    let mut read = 0;
    while read < buffer.len() {
        match reader.read(&mut buffer[read..]) {
            Ok(0) if read == 0 => return Ok(ReadState::Eof),
            Ok(0) => return Ok(ReadState::Truncated),
            Ok(count) => read += count,
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(ReadState::Complete)
}

fn encode_upsert(sequence: u64, points: &[Point]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.push(1);
    output.extend_from_slice(&sequence.to_le_bytes());
    put_count(&mut output, points.len())?;
    for point in points {
        put_string(&mut output, &point.id)?;
        put_count(&mut output, point.vector.len())?;
        for value in &point.vector {
            output.extend_from_slice(&value.to_le_bytes());
        }
        put_count(&mut output, point.metadata.len())?;
        for (key, value) in &point.metadata {
            put_string(&mut output, key)?;
            encode_value(&mut output, value)?;
        }
    }
    Ok(output)
}

fn encode_delete(sequence: u64, ids: &[String]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output.push(2);
    output.extend_from_slice(&sequence.to_le_bytes());
    put_count(&mut output, ids.len())?;
    for id in ids {
        put_string(&mut output, id)?;
    }
    Ok(output)
}

fn apply_payload(collection: &mut Collection, payload: &[u8]) -> Result<Vec<String>> {
    let mut decoder = Decoder::new(payload);
    let operation = decoder.byte()?;
    let sequence = decoder.u64()?;
    // A manifest can become durable immediately before the WAL is truncated.
    // Replaying that crash state must ignore records already in the snapshot.
    if sequence <= collection.latest_sequence() {
        return Ok(Vec::new());
    }
    let expected = collection
        .latest_sequence()
        .checked_add(1)
        .ok_or(Error::SequenceOverflow)?;
    if sequence != expected {
        return Err(Error::CorruptStorage(format!(
            "non-consecutive WAL sequence: expected {expected}, got {sequence}"
        )));
    }
    match operation {
        1 => {
            let count = decoder.count()?;
            let mut points = Vec::with_capacity(count);
            for _ in 0..count {
                let id = decoder.string()?;
                let dimension = decoder.count()?;
                if dimension != collection.config().dimension {
                    return Err(Error::CorruptStorage(format!(
                        "stored vector dimension {dimension} differs from collection dimension {}",
                        collection.config().dimension
                    )));
                }
                let mut vector = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    vector.push(decoder.f32()?);
                }
                let metadata_count = decoder.count()?;
                let mut metadata = BTreeMap::new();
                for _ in 0..metadata_count {
                    metadata.insert(decoder.string()?, decoder.value()?);
                }
                points.push(UpsertPoint {
                    id,
                    vector,
                    metadata,
                });
            }
            decoder.finish()?;
            let changed_ids = points.iter().map(|point| point.id.clone()).collect();
            let prepared = collection.prepare_upsert(points, sequence)?;
            collection.apply_prepared_upsert(prepared, sequence);
            Ok(changed_ids)
        }
        2 => {
            let count = decoder.count()?;
            if count == 0 {
                return Err(Error::CorruptStorage("empty delete WAL record".into()));
            }
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                ids.push(decoder.string()?);
            }
            decoder.finish()?;
            collection.apply_delete_at(&ids, sequence);
            Ok(ids)
        }
        tag => Err(Error::CorruptStorage(format!(
            "unknown WAL operation {tag}"
        ))),
    }
}

fn put_count(output: &mut Vec<u8>, count: usize) -> Result<()> {
    let count = u32::try_from(count).map_err(|_| Error::InvalidQuery("batch is too large"))?;
    output.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

fn put_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    put_count(output, value.len())?;
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn encode_value(output: &mut Vec<u8>, value: &Value) -> Result<()> {
    match value {
        Value::Keyword(value) => {
            output.push(1);
            put_string(output, value)?;
        }
        Value::Integer(value) => {
            output.push(2);
            output.extend_from_slice(&value.to_le_bytes());
        }
        Value::Float(value) if value.is_finite() => {
            output.push(3);
            output.extend_from_slice(&value.to_le_bytes());
        }
        Value::Float(_) => return Err(Error::InvalidQuery("metadata floats must be finite")),
        Value::Boolean(value) => {
            output.push(4);
            output.push(u8::from(*value));
        }
    }
    Ok(())
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::CorruptStorage("WAL length overflow".into()))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::CorruptStorage("truncated WAL payload".into()))?;
        self.position = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed slice"),
        ))
    }

    fn f32(&mut self) -> Result<f32> {
        let value = f32::from_le_bytes(self.take(4)?.try_into().expect("fixed slice"));
        if !value.is_finite() {
            return Err(Error::CorruptStorage("non-finite vector component".into()));
        }
        Ok(value)
    }

    fn count(&mut self) -> Result<usize> {
        let count = self.u32()? as usize;
        if count > MAX_ITEMS {
            return Err(Error::CorruptStorage("WAL item count exceeds limit".into()));
        }
        Ok(count)
    }

    fn string(&mut self) -> Result<String> {
        let length = self.count()?;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::CorruptStorage("WAL contains invalid UTF-8".into()))
    }

    fn value(&mut self) -> Result<Value> {
        match self.byte()? {
            1 => Ok(Value::Keyword(self.string()?)),
            2 => Ok(Value::Integer(i64::from_le_bytes(
                self.take(8)?.try_into().expect("fixed slice"),
            ))),
            3 => {
                let value = f64::from_le_bytes(self.take(8)?.try_into().expect("fixed slice"));
                if !value.is_finite() {
                    return Err(Error::CorruptStorage("non-finite metadata float".into()));
                }
                Ok(Value::Float(value))
            }
            4 => match self.byte()? {
                0 => Ok(Value::Boolean(false)),
                1 => Ok(Value::Boolean(true)),
                _ => Err(Error::CorruptStorage("invalid boolean encoding".into())),
            },
            tag => Err(Error::CorruptStorage(format!(
                "unknown metadata value tag {tag}"
            ))),
        }
    }

    fn finish(self) -> Result<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::CorruptStorage(
                "trailing bytes in WAL payload".into(),
            ))
        }
    }
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
        _ => Err(Error::CorruptStorage(format!("unknown metric tag {tag}"))),
    }
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn config() -> CollectionConfig {
        CollectionConfig {
            dimension: 2,
            metric: Metric::Cosine,
        }
    }

    fn point(id: &str, vector: [f32; 2]) -> UpsertPoint {
        UpsertPoint {
            id: id.into(),
            vector: vector.into(),
            metadata: BTreeMap::from([("kind".into(), Value::Keyword("test".into()))]),
        }
    }

    #[test]
    fn committed_mutations_survive_restart() {
        let directory = test_directory("restart");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection
                .upsert(vec![point("keep", [1.0, 0.0]), point("remove", [0.0, 1.0])])
                .unwrap();
            collection.delete(vec!["remove".into()]).unwrap();
        }
        let collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(collection.latest_sequence(), 2);
        assert!(collection.collection().get("keep").is_some());
        assert!(collection.collection().get("remove").is_none());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn incomplete_final_frame_is_discarded() {
        let directory = test_directory("truncated");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("safe", [1.0, 0.0])]).unwrap();
        }
        let wal_path = directory.join("write.wal");
        let valid_length = fs::metadata(&wal_path).unwrap().len();
        OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap()
            .write_all(b"FVWL\x40\x00")
            .unwrap();

        let collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(collection.latest_sequence(), 1);
        assert_eq!(fs::metadata(wal_path).unwrap().len(), valid_length);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn checksum_corruption_is_rejected() {
        let directory = test_directory("corrupt");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("p", [1.0, 0.0])]).unwrap();
        }
        let wal_path = directory.join("write.wal");
        let mut wal = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .unwrap();
        wal.seek(SeekFrom::Start(12)).unwrap();
        wal.write_all(&[0xff]).unwrap();
        wal.sync_all().unwrap();

        let error = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap_err();
        assert!(matches!(error, Error::CorruptStorage(_)));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn persisted_configuration_cannot_change() {
        let directory = test_directory("config");
        drop(PersistentCollection::open(&directory, config(), Durability::Sync).unwrap());
        let error = PersistentCollection::open(
            &directory,
            CollectionConfig {
                dimension: 3,
                metric: Metric::Cosine,
            },
            Durability::Sync,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidConfig(_)));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_a_second_writer_for_the_same_directory() {
        let directory = test_directory("exclusive-lock");
        let first = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        let error = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap_err();
        assert!(matches!(error, Error::Concurrency(_)));
        drop(first);
        assert!(PersistentCollection::open(&directory, config(), Durability::Sync).is_ok());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn crc32c_matches_standard_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn flush_checkpoints_wal_and_recovers_segment() {
        let directory = test_directory("segment-restart");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection
                .upsert(vec![point("a", [1.0, 0.0]), point("b", [0.0, 1.0])])
                .unwrap();
            assert!(!collection.has_approximate_index());
            assert_eq!(collection.flush().unwrap(), 1);
            assert!(collection.has_approximate_index());
            assert_eq!(fs::metadata(directory.join("write.wal")).unwrap().len(), 0);
        }

        let collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(collection.latest_sequence(), 1);
        assert_eq!(collection.collection().len(), 2);
        assert!(collection.has_approximate_index());
        let hits = collection
            .search_with_ef(vec![1.0, 0.0], 2, None, 16)
            .unwrap();
        assert_eq!(hits[0].id, "a");
        assert!(directory.join("segment-00000000000000000001.fvs").exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn mutations_after_flush_replay_over_segment() {
        let directory = test_directory("segment-and-wal");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("a", [1.0, 0.0])]).unwrap();
            collection.flush().unwrap();
            assert!(collection.has_approximate_index());
            collection.upsert(vec![point("b", [0.0, 1.0])]).unwrap();
            assert!(collection.has_approximate_index());
            collection.delete(vec!["a".into()]).unwrap();
            let hits = collection.search(vec![0.0, 1.0], 2, None).unwrap();
            assert_eq!(
                hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
                ["b"]
            );
        }

        let collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(collection.latest_sequence(), 3);
        assert!(collection.has_approximate_index());
        assert!(collection.collection().get("a").is_none());
        assert!(collection.collection().get("b").is_some());
        let hits = collection.search(vec![0.0, 1.0], 2, None).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["b"]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn repeated_flush_replaces_obsolete_segment() {
        let directory = test_directory("segment-replace");
        let first_segment = directory.join("segment-00000000000000000001.fvs");
        let second_segment = directory.join("segment-00000000000000000002.fvs");
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("a", [1.0, 0.0])]).unwrap();
            collection.flush().unwrap();
            collection.upsert(vec![point("b", [0.0, 1.0])]).unwrap();
            collection.flush().unwrap();
        }

        assert!(!first_segment.exists());
        assert!(second_segment.exists());
        let collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(collection.collection().len(), 2);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn segment_corruption_is_rejected() {
        let directory = test_directory("segment-corrupt");
        let segment_path;
        {
            let mut collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("a", [1.0, 0.0])]).unwrap();
            collection.flush().unwrap();
            segment_path = directory.join("segment-00000000000000000001.fvs");
        }
        let mut segment = OpenOptions::new()
            .read(true)
            .write(true)
            .open(segment_path)
            .unwrap();
        segment.seek(SeekFrom::Start(20)).unwrap();
        segment.write_all(&[0xff]).unwrap();
        segment.sync_all().unwrap();

        let error = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap_err();
        assert!(matches!(error, Error::CorruptStorage(_)));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn filtered_search_stays_correct_with_loaded_graph() {
        let directory = test_directory("filtered-graph");
        let mut collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        collection
            .upsert(vec![point("a", [1.0, 0.0]), point("b", [0.9, 0.1])])
            .unwrap();
        collection.flush().unwrap();
        let filter = Filter::Eq {
            field: "kind".into(),
            value: Value::Keyword("does-not-match".into()),
        };
        assert!(
            collection
                .search_with_ef(vec![1.0, 0.0], 2, Some(&filter), 16)
                .unwrap()
                .is_empty()
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn updated_point_does_not_leak_stale_graph_vector() {
        let directory = test_directory("updated-graph-point");
        let mut collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        collection
            .upsert(vec![point("a", [1.0, 0.0]), point("b", [0.0, 1.0])])
            .unwrap();
        collection.flush().unwrap();
        collection.upsert(vec![point("a", [-1.0, 0.0])]).unwrap();

        let hits = collection.search(vec![1.0, 0.0], 2, None).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["b", "a"]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn metadata_index_merges_changed_payloads_correctly() {
        let directory = test_directory("metadata-delta");
        let mut collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        let tagged = |id: &str, tenant: &str, vector: [f32; 2]| UpsertPoint {
            id: id.into(),
            vector: vector.into(),
            metadata: BTreeMap::from([
                ("tenant".into(), Value::Keyword(tenant.into())),
                (
                    "price".into(),
                    Value::Integer(if id == "a" { 10 } else { 20 }),
                ),
            ]),
        };
        collection
            .upsert(vec![
                tagged("a", "one", [1.0, 0.0]),
                tagged("b", "two", [0.0, 1.0]),
            ])
            .unwrap();
        collection.flush().unwrap();
        collection
            .upsert(vec![tagged("a", "two", [1.0, 0.0])])
            .unwrap();

        let old_tenant = Filter::Eq {
            field: "tenant".into(),
            value: Value::Keyword("one".into()),
        };
        assert!(
            collection
                .search(vec![1.0, 0.0], 10, Some(&old_tenant))
                .unwrap()
                .is_empty()
        );

        let new_tenant_with_range = Filter::And(vec![
            Filter::Eq {
                field: "tenant".into(),
                value: Value::Keyword("two".into()),
            },
            Filter::Range {
                field: "price".into(),
                gte: Some(10.0),
                lt: Some(21.0),
            },
        ]);
        let hits = collection
            .search(vec![1.0, 0.0], 10, Some(&new_tenant_with_range))
            .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn flush_snapshot_preserves_writes_that_arrive_during_build() {
        let directory = test_directory("flush-race");
        let mut collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        collection.upsert(vec![point("a", [1.0, 0.0])]).unwrap();
        let snapshot = collection.flush_snapshot();
        collection.upsert(vec![point("b", [0.0, 1.0])]).unwrap();
        let prepared = PersistentCollection::build_flush(snapshot).unwrap();
        assert_eq!(collection.publish_flush(prepared).unwrap(), 1);
        assert!(fs::metadata(directory.join("write.wal")).unwrap().len() > 0);
        drop(collection);

        let reopened = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(reopened.latest_sequence(), 2);
        assert!(reopened.collection().get("a").is_some());
        assert!(reopened.collection().get("b").is_some());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stale_flush_cannot_replace_a_newer_manifest() {
        let directory = test_directory("stale-flush");
        let mut collection =
            PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        collection.upsert(vec![point("a", [1.0, 0.0])]).unwrap();
        let stale = PersistentCollection::build_flush(collection.flush_snapshot()).unwrap();
        collection.upsert(vec![point("b", [0.0, 1.0])]).unwrap();
        collection.flush().unwrap();
        assert_eq!(collection.publish_flush(stale).unwrap(), 2);
        drop(collection);

        let reopened = PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(reopened.latest_sequence(), 2);
        assert_eq!(reopened.collection().len(), 2);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn malformed_segment_inputs_never_panic() {
        let mut state = 0x8bad_f00d_dead_beef_u64;
        for length in 0..2_048 {
            let mut bytes = vec![0_u8; length];
            for byte in &mut bytes {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *byte = (state >> 32) as u8;
            }
            let outcome = std::panic::catch_unwind(|| decode_segment(&bytes));
            assert!(outcome.is_ok(), "decoder panicked for {length} bytes");
        }
    }

    #[test]
    fn every_torn_wal_suffix_recovers_to_an_atomic_boundary() {
        let source = test_directory("wal-prefix-source");
        {
            let mut collection =
                PersistentCollection::open(&source, config(), Durability::Sync).unwrap();
            collection.upsert(vec![point("safe", [1.0, 0.0])]).unwrap();
        }
        let wal = fs::read(source.join("write.wal")).unwrap();
        let metadata = fs::read(source.join("collection.meta")).unwrap();
        fs::remove_dir_all(source).unwrap();

        for cut in 0..=wal.len() {
            let directory = test_directory(&format!("wal-prefix-{cut}"));
            fs::create_dir(&directory).unwrap();
            fs::write(directory.join("collection.meta"), &metadata).unwrap();
            fs::write(directory.join("write.wal"), &wal[..cut]).unwrap();
            let collection =
                PersistentCollection::open(&directory, config(), Durability::Sync).unwrap();
            assert_eq!(
                collection.latest_sequence(),
                u64::from(cut == wal.len()),
                "unexpected sequence for WAL prefix {cut}"
            );
            drop(collection);
            fs::remove_dir_all(directory).unwrap();
        }
    }
}
