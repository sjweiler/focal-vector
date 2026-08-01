use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openraft::storage::{RaftSnapshotBuilder, RaftStateMachine, Snapshot};
use openraft::{
    Entry, EntryPayload, LogId, Membership, OptionalSend, SnapshotMeta, StorageError,
    StorageIOError, StoredMembership,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::metadata_index::MetadataIndex;
use crate::raft_storage::{FocalRaftConfig, NodeId, ShardCommand, ShardResponse};
use crate::{
    Collection, CollectionConfig, Error, Filter, HnswConfig, HnswIndex, Result as FocalResult,
    SearchHit,
};

const STATE_MAGIC: &[u8; 4] = b"FVRS";
const STATE_VERSION: u8 = 1;
const MAX_STATE_RECORD_BYTES: usize = 256 * 1024 * 1024;
const SNAPSHOT_MAGIC: &[u8; 4] = b"FVSS";
const SNAPSHOT_VERSION: u8 = 1;
const MIN_REINDEXED_POINTS: usize = 256;
const REINDEX_DIRTY_DIVISOR: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AppliedPayload {
    Blank,
    Command {
        command: ShardCommand,
        response: ShardResponse,
    },
    Membership(Membership<NodeId, openraft::BasicNode>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppliedRecord {
    log_id: LogId<NodeId>,
    payload: AppliedPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateSnapshot {
    config: CollectionConfig,
    points: Vec<crate::Point>,
    collection_sequence: u64,
    last_applied: Option<LogId<NodeId>>,
    membership: StoredMembership<NodeId, openraft::BasicNode>,
    dedup: BTreeMap<String, BTreeMap<u64, ShardResponse>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSnapshot {
    meta: SnapshotMeta<NodeId, openraft::BasicNode>,
    data: Vec<u8>,
    #[serde(default)]
    checksum: Option<u32>,
}

#[derive(Debug)]
struct StateInner {
    journal: File,
    collection: Collection,
    index: Option<HnswIndex>,
    metadata_index: Option<MetadataIndex>,
    dirty_ids: HashSet<String>,
    last_applied: Option<LogId<NodeId>>,
    membership: StoredMembership<NodeId, openraft::BasicNode>,
    dedup: BTreeMap<String, BTreeMap<u64, ShardResponse>>,
    current_snapshot: Option<StoredSnapshot>,
}

#[derive(Debug, Clone)]
pub struct DurableShardStateMachine {
    directory: Arc<PathBuf>,
    config: CollectionConfig,
    inner: Arc<Mutex<StateInner>>,
    index_refresh: Arc<Mutex<()>>,
}

impl DurableShardStateMachine {
    pub fn open(directory: impl AsRef<Path>, config: CollectionConfig) -> FocalResult<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let mut collection = Collection::new(config)?;
        let mut last_applied = None;
        let mut membership = StoredMembership::default();
        let mut dedup = BTreeMap::new();
        let mut index = None;
        let mut metadata_index = None;
        let mut dirty_ids = HashSet::new();
        let current_snapshot = read_snapshot(&directory)?;
        if let Some(stored) = &current_snapshot {
            let snapshot = decode_state_snapshot(&stored.data)?;
            if snapshot.config != config {
                return Err(Error::CorruptStorage(
                    "Raft snapshot collection configuration mismatch".into(),
                ));
            }
            let indexes = build_indexes(config, &snapshot.points)?;
            index = Some(indexes.0);
            metadata_index = Some(indexes.1);
            collection.restore_snapshot(snapshot.points, snapshot.collection_sequence)?;
            last_applied = snapshot.last_applied;
            membership = snapshot.membership;
            dedup = snapshot.dedup;
        }

        let mut journal = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(directory.join("state.journal"))?;
        let records: Vec<AppliedRecord> = recover_records(&mut journal, STATE_MAGIC)?;
        for record in records {
            if last_applied.is_some_and(|applied| record.log_id.index <= applied.index) {
                continue;
            }
            replay_record(
                &mut collection,
                &mut last_applied,
                &mut membership,
                &mut dedup,
                &mut dirty_ids,
                record,
            )?;
        }
        journal.seek(SeekFrom::End(0))?;
        Ok(Self {
            directory: Arc::new(directory),
            config,
            inner: Arc::new(Mutex::new(StateInner {
                journal,
                collection,
                index,
                metadata_index,
                dirty_ids,
                last_applied,
                membership,
                dedup,
                current_snapshot,
            })),
            index_refresh: Arc::new(Mutex::new(())),
        })
    }

    async fn refresh_index_if_needed(&self) -> FocalResult<()> {
        let _refresh = self.index_refresh.lock().await;
        let rebuild = {
            let inner = self.inner.lock().await;
            should_rebuild_index(
                inner.index.as_ref(),
                inner.dirty_ids.len(),
                inner.collection.len(),
            )
            .then(|| {
                (
                    inner.collection.latest_sequence(),
                    inner.collection.snapshot_points(),
                )
            })
        };
        let Some((sequence, points)) = rebuild else {
            return Ok(());
        };
        let config = self.config;
        let (index, metadata_index) =
            tokio::task::spawn_blocking(move || build_indexes(config, &points))
                .await
                .map_err(|error| {
                    Error::Concurrency(format!("HNSW index task failed: {error}"))
                })??;
        let mut inner = self.inner.lock().await;
        if inner.collection.latest_sequence() == sequence {
            inner.index = Some(index);
            inner.metadata_index = Some(metadata_index);
            inner.dirty_ids.clear();
        }
        Ok(())
    }

    pub async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> FocalResult<Vec<SearchHit>> {
        self.search_with_ef(query, k, filter, k.saturating_mul(16).max(256))
            .await
    }

    pub async fn search_with_ef(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
        ef_search: usize,
    ) -> FocalResult<Vec<SearchHit>> {
        if ef_search < k {
            return Err(Error::InvalidQuery("ef_search must be at least k"));
        }
        self.refresh_index_if_needed().await?;
        let inner = self.inner.lock().await;
        let filter = filter.cloned();
        let Some(index) = &inner.index else {
            return inner.collection.search(query, k, filter.as_ref());
        };
        let mut filtered_candidates = None;
        if let Some(filter) = &filter {
            let Some(metadata_index) = &inner.metadata_index else {
                return inner.collection.search(query, k, Some(filter));
            };
            let mut candidates = metadata_index.candidates(filter);
            candidates.retain(|id| !inner.dirty_ids.contains(id));
            let use_exact = candidates.len().saturating_mul(4) < index.len()
                || candidates.len() <= k.saturating_mul(16).max(256);
            if use_exact {
                let mut hits =
                    inner
                        .collection
                        .search_ids(query.clone(), k, &candidates, Some(filter))?;
                hits.extend(inner.collection.search_ids(
                    query,
                    k,
                    &inner.dirty_ids,
                    Some(filter),
                )?);
                return Ok(merge_hits(hits, k));
            }
            filtered_candidates = Some(candidates);
        }
        let mut graph_k = if filter.is_some() {
            ef_search
                .max(k.saturating_mul(8))
                .saturating_add(inner.dirty_ids.len())
                .min(index.len())
        } else {
            k.saturating_add(inner.dirty_ids.len()).min(index.len())
        };
        let mut hits = Vec::with_capacity(graph_k.saturating_add(k));
        while graph_k > 0 {
            hits.clear();
            for hit in index.search(query.clone(), graph_k, ef_search.max(graph_k))? {
                if inner.dirty_ids.contains(&hit.id) {
                    continue;
                }
                if let Some(filter) = &filter {
                    let Some(point) = inner.collection.get(&hit.id) else {
                        return Err(Error::CorruptStorage(format!(
                            "HNSW point {} is missing from the Raft state machine",
                            hit.id
                        )));
                    };
                    if !filter.matches(&point.metadata) {
                        continue;
                    }
                }
                let point = inner.collection.get(&hit.id).ok_or_else(|| {
                    Error::CorruptStorage(format!(
                        "HNSW point {} is missing from the Raft state machine",
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
            if filter.is_none() || hits.len() >= k || graph_k == index.len() {
                break;
            }
            graph_k = graph_k.saturating_mul(2).min(index.len());
        }
        if hits.len() < k
            && let (Some(filter), Some(candidates)) = (&filter, &filtered_candidates)
        {
            hits.extend(
                inner
                    .collection
                    .search_ids(query.clone(), k, candidates, Some(filter))?,
            );
        }
        hits.extend(
            inner
                .collection
                .search_ids(query, k, &inner.dirty_ids, filter.as_ref())?,
        );
        Ok(merge_hits(hits, k))
    }

    pub async fn has_approximate_index(&self) -> bool {
        self.inner.lock().await.index.is_some()
    }

    pub async fn pending_point_count(&self) -> usize {
        self.inner.lock().await.dirty_ids.len()
    }

    pub async fn applied_index(&self) -> Option<u64> {
        self.inner
            .lock()
            .await
            .last_applied
            .map(|log_id| log_id.index)
    }

    pub async fn len(&self) -> usize {
        self.inner.lock().await.collection.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.collection.is_empty()
    }

    fn snapshot_path(&self) -> PathBuf {
        self.directory.join("state.snapshot")
    }
}

impl RaftStateMachine<FocalRaftConfig> for DurableShardStateMachine {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<LogId<NodeId>>,
            StoredMembership<NodeId, openraft::BasicNode>,
        ),
        StorageError<NodeId>,
    > {
        let inner = self.inner.lock().await;
        Ok((inner.last_applied, inner.membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<ShardResponse>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<FocalRaftConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut inner = self.inner.lock().await;
        let mut responses = Vec::new();
        for entry in entries {
            let (payload, response) = prepare_payload(&inner, &entry);
            let record = AppliedRecord {
                log_id: entry.log_id,
                payload,
            };
            append_record(&mut inner.journal, STATE_MAGIC, &record)
                .map_err(|error| StorageIOError::write_state_machine(&error))?;
            replay_record_inner(&mut inner, record)
                .map_err(|error| StorageIOError::write_state_machine(&error))?;
            responses.push(response);
        }
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, openraft::BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let data = snapshot.into_inner();
        let decoded = decode_state_snapshot(&data)
            .map_err(|error| StorageIOError::read_snapshot(Some(meta.signature()), &error))?;
        if decoded.config != self.config {
            let error = std::io::Error::new(ErrorKind::InvalidData, "snapshot config mismatch");
            return Err(StorageIOError::read_snapshot(Some(meta.signature()), &error).into());
        }
        let checksum = stored_snapshot_checksum(meta, &data)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;
        let stored = StoredSnapshot {
            meta: meta.clone(),
            data,
            checksum: Some(checksum),
        };
        write_snapshot(&self.snapshot_path(), &stored)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;

        let (index, metadata_index) = build_indexes(decoded.config, &decoded.points)
            .map_err(|error| StorageIOError::read_state_machine(&error))?;
        let mut collection = Collection::new(decoded.config)
            .map_err(|error| StorageIOError::read_state_machine(&error))?;
        collection
            .restore_snapshot(decoded.points, decoded.collection_sequence)
            .map_err(|error| StorageIOError::read_state_machine(&error))?;
        let mut inner = self.inner.lock().await;
        inner
            .journal
            .set_len(0)
            .map_err(|error| StorageIOError::write_state_machine(&error))?;
        inner
            .journal
            .seek(SeekFrom::Start(0))
            .map_err(|error| StorageIOError::write_state_machine(&error))?;
        inner
            .journal
            .sync_data()
            .map_err(|error| StorageIOError::write_state_machine(&error))?;
        inner.collection = collection;
        inner.index = Some(index);
        inner.metadata_index = Some(metadata_index);
        inner.dirty_ids.clear();
        inner.last_applied = decoded.last_applied;
        inner.membership = decoded.membership;
        inner.dedup = decoded.dedup;
        inner.current_snapshot = Some(stored);
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<FocalRaftConfig>>, StorageError<NodeId>> {
        Ok(self
            .inner
            .lock()
            .await
            .current_snapshot
            .clone()
            .map(|stored| Snapshot {
                meta: stored.meta,
                snapshot: Box::new(Cursor::new(stored.data)),
            }))
    }
}

impl RaftSnapshotBuilder<FocalRaftConfig> for DurableShardStateMachine {
    async fn build_snapshot(&mut self) -> Result<Snapshot<FocalRaftConfig>, StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        let snapshot = StateSnapshot {
            config: self.config,
            points: inner.collection.snapshot_points(),
            collection_sequence: inner.collection.latest_sequence(),
            last_applied: inner.last_applied,
            membership: inner.membership.clone(),
            dedup: inner.dedup.clone(),
        };
        let (index, metadata_index) = build_indexes(self.config, &snapshot.points)
            .map_err(|error| StorageIOError::write_snapshot(None, &error))?;
        let data = encode_state_snapshot(&snapshot)
            .map_err(|error| StorageIOError::write_snapshot(None, &error))?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| StorageIOError::write_snapshot(None, &error))?
            .as_nanos();
        let meta = SnapshotMeta {
            last_log_id: inner.last_applied,
            last_membership: inner.membership.clone(),
            snapshot_id: format!("{}-{nonce}", inner.last_applied.map_or(0, |log| log.index)),
        };
        let checksum = stored_snapshot_checksum(&meta, &data)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;
        let stored = StoredSnapshot {
            meta: meta.clone(),
            data: data.clone(),
            checksum: Some(checksum),
        };
        write_snapshot(&self.snapshot_path(), &stored)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;
        inner.current_snapshot = Some(stored);
        inner.index = Some(index);
        inner.metadata_index = Some(metadata_index);
        inner.dirty_ids.clear();
        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(data)),
        })
    }
}

fn prepare_payload(
    inner: &StateInner,
    entry: &Entry<FocalRaftConfig>,
) -> (AppliedPayload, ShardResponse) {
    match &entry.payload {
        EntryPayload::Blank => (
            AppliedPayload::Blank,
            ShardResponse::Applied {
                sequence: inner.collection.latest_sequence(),
            },
        ),
        EntryPayload::Membership(membership) => (
            AppliedPayload::Membership(membership.clone()),
            ShardResponse::Applied {
                sequence: inner.collection.latest_sequence(),
            },
        ),
        EntryPayload::Normal(command) => {
            let (client_id, request_id) = command_identity(command);
            if let Some(response) = inner
                .dedup
                .get(client_id)
                .and_then(|requests| requests.get(&request_id))
                .cloned()
            {
                return (
                    AppliedPayload::Command {
                        command: command.clone(),
                        response: response.clone(),
                    },
                    response,
                );
            }
            let sequence = inner.collection.latest_sequence().checked_add(1);
            let response = match (command, sequence) {
                (_, None) => ShardResponse::Rejected {
                    message: "collection sequence overflow".into(),
                },
                (ShardCommand::Upsert { points, .. }, Some(sequence)) => inner
                    .collection
                    .prepare_upsert(points.clone(), sequence)
                    .map(|_| ShardResponse::Applied { sequence })
                    .unwrap_or_else(|error| ShardResponse::Rejected {
                        message: error.to_string(),
                    }),
                (ShardCommand::Delete { ids, .. }, Some(_))
                    if ids.is_empty() || ids.iter().any(String::is_empty) =>
                {
                    ShardResponse::Rejected {
                        message: "delete IDs must be non-empty".into(),
                    }
                }
                (ShardCommand::Delete { .. }, Some(sequence)) => {
                    ShardResponse::Applied { sequence }
                }
            };
            (
                AppliedPayload::Command {
                    command: command.clone(),
                    response: response.clone(),
                },
                response,
            )
        }
    }
}

fn command_identity(command: &ShardCommand) -> (&str, u64) {
    match command {
        ShardCommand::Upsert {
            client_id,
            request_id,
            ..
        }
        | ShardCommand::Delete {
            client_id,
            request_id,
            ..
        } => (client_id, *request_id),
    }
}

fn replay_record_inner(inner: &mut StateInner, record: AppliedRecord) -> FocalResult<()> {
    replay_record(
        &mut inner.collection,
        &mut inner.last_applied,
        &mut inner.membership,
        &mut inner.dedup,
        &mut inner.dirty_ids,
        record,
    )
}

fn replay_record(
    collection: &mut Collection,
    last_applied: &mut Option<LogId<NodeId>>,
    membership: &mut StoredMembership<NodeId, openraft::BasicNode>,
    dedup: &mut BTreeMap<String, BTreeMap<u64, ShardResponse>>,
    dirty_ids: &mut HashSet<String>,
    record: AppliedRecord,
) -> FocalResult<()> {
    match record.payload {
        AppliedPayload::Blank => {}
        AppliedPayload::Membership(value) => {
            *membership = StoredMembership::new(Some(record.log_id), value)
        }
        AppliedPayload::Command { command, response } => {
            let (client_id, request_id) = command_identity(&command);
            let duplicate = dedup
                .get(client_id)
                .is_some_and(|requests| requests.contains_key(&request_id));
            if !duplicate {
                if let ShardResponse::Applied { sequence } = response {
                    match &command {
                        ShardCommand::Upsert { points, .. } => {
                            dirty_ids.extend(points.iter().map(|point| point.id.clone()));
                            let prepared = collection.prepare_upsert(points.clone(), sequence)?;
                            collection.apply_prepared_upsert(prepared, sequence);
                        }
                        ShardCommand::Delete { ids, .. } => {
                            dirty_ids.extend(ids.iter().cloned());
                            collection.apply_delete_at(ids, sequence)
                        }
                    }
                }
                dedup
                    .entry(client_id.to_owned())
                    .or_default()
                    .insert(request_id, response);
            }
        }
    }
    *last_applied = Some(record.log_id);
    Ok(())
}

fn build_indexes(
    config: CollectionConfig,
    points: &[crate::Point],
) -> FocalResult<(HnswIndex, MetadataIndex)> {
    let index = HnswIndex::build(
        config.dimension,
        config.metric,
        HnswConfig {
            m: 32,
            ef_construction: 400,
        },
        points
            .iter()
            .map(|point| (point.id.clone(), point.vector.clone())),
    )?;
    Ok((index, MetadataIndex::build(points)))
}

fn encode_state_snapshot(snapshot: &StateSnapshot) -> FocalResult<Vec<u8>> {
    let payload =
        serde_json::to_vec(snapshot).map_err(|error| Error::CorruptStorage(error.to_string()))?;
    let length = u64::try_from(payload.len())
        .map_err(|_| Error::CorruptStorage("Raft snapshot is too large".into()))?;
    let mut encoded = Vec::with_capacity(17_usize.saturating_add(payload.len()));
    encoded.extend_from_slice(SNAPSHOT_MAGIC);
    encoded.push(SNAPSHOT_VERSION);
    encoded.extend_from_slice(&length.to_le_bytes());
    encoded.extend_from_slice(&payload);
    encoded.extend_from_slice(&crc32c(&payload).to_le_bytes());
    Ok(encoded)
}

fn decode_state_snapshot(bytes: &[u8]) -> FocalResult<StateSnapshot> {
    if !bytes.starts_with(SNAPSHOT_MAGIC) {
        return serde_json::from_slice(bytes)
            .map_err(|error| Error::CorruptStorage(error.to_string()));
    }
    if bytes.len() < 17 || bytes[4] != SNAPSHOT_VERSION {
        return Err(Error::CorruptStorage("invalid Raft snapshot header".into()));
    }
    let length = u64::from_le_bytes(
        bytes[5..13]
            .try_into()
            .expect("snapshot length is eight bytes"),
    );
    let length = usize::try_from(length)
        .map_err(|_| Error::CorruptStorage("Raft snapshot length is too large".into()))?;
    let expected = 17_usize
        .checked_add(length)
        .ok_or_else(|| Error::CorruptStorage("Raft snapshot length overflow".into()))?;
    if bytes.len() != expected {
        return Err(Error::CorruptStorage("truncated Raft snapshot".into()));
    }
    let payload = &bytes[13..13 + length];
    let stored = u32::from_le_bytes(
        bytes[13 + length..expected]
            .try_into()
            .expect("snapshot checksum is four bytes"),
    );
    if crc32c(payload) != stored {
        return Err(Error::CorruptStorage(
            "Raft snapshot checksum mismatch".into(),
        ));
    }
    serde_json::from_slice(payload).map_err(|error| Error::CorruptStorage(error.to_string()))
}

fn should_rebuild_index(index: Option<&HnswIndex>, dirty_count: usize, point_count: usize) -> bool {
    if dirty_count < MIN_REINDEXED_POINTS {
        return false;
    }
    index.is_none()
        || dirty_count
            >= point_count
                .div_ceil(REINDEX_DIRTY_DIVISOR)
                .max(MIN_REINDEXED_POINTS)
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

fn append_record<T: Serialize>(
    file: &mut File,
    magic: &[u8; 4],
    record: &T,
) -> std::io::Result<()> {
    let payload = serde_json::to_vec(record).map_err(invalid_data)?;
    if payload.len() > MAX_STATE_RECORD_BYTES {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "state record exceeds size limit",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "state record is too large"))?;
    let start = file.seek(SeekFrom::End(0))?;
    let result = (|| {
        file.write_all(magic)?;
        file.write_all(&[STATE_VERSION])?;
        file.write_all(&length.to_le_bytes())?;
        file.write_all(&payload)?;
        file.write_all(&crc32c(&payload).to_le_bytes())?;
        file.sync_data()
    })();
    if result.is_err() {
        let _ = file.set_len(start);
        let _ = file.seek(SeekFrom::End(0));
    }
    result
}

fn recover_records<T: for<'de> Deserialize<'de>>(
    file: &mut File,
    magic: &[u8; 4],
) -> std::io::Result<Vec<T>> {
    file.seek(SeekFrom::Start(0))?;
    let mut records = Vec::new();
    loop {
        let start = file.stream_position()?;
        let mut header = [0_u8; 9];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                file.set_len(start)?;
                file.sync_data()?;
                break;
            }
            Err(error) => return Err(error),
        }
        if &header[..4] != magic || header[4] != STATE_VERSION {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "invalid state journal header",
            ));
        }
        let length = u32::from_le_bytes(header[5..9].try_into().expect("four bytes")) as usize;
        if length > MAX_STATE_RECORD_BYTES {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "state record exceeds size limit",
            ));
        }
        let mut frame = vec![0_u8; length + 4];
        if let Err(error) = file.read_exact(&mut frame) {
            if error.kind() == ErrorKind::UnexpectedEof {
                file.set_len(start)?;
                file.sync_data()?;
                break;
            }
            return Err(error);
        }
        let stored = u32::from_le_bytes(frame[length..].try_into().expect("four bytes"));
        if crc32c(&frame[..length]) != stored {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "state journal checksum mismatch",
            ));
        }
        records.push(serde_json::from_slice(&frame[..length]).map_err(invalid_data)?);
    }
    Ok(records)
}

fn write_snapshot(path: &Path, snapshot: &StoredSnapshot) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(snapshot).map_err(invalid_data)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    File::open(path.parent().expect("snapshot has parent"))?.sync_all()
}

fn read_snapshot(directory: &Path) -> FocalResult<Option<StoredSnapshot>> {
    let path = directory.join("state.snapshot");
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let snapshot: StoredSnapshot =
        serde_json::from_slice(&bytes).map_err(|error| Error::CorruptStorage(error.to_string()))?;
    if let Some(stored) = snapshot.checksum {
        let actual = stored_snapshot_checksum(&snapshot.meta, &snapshot.data)
            .map_err(|error| Error::CorruptStorage(error.to_string()))?;
        if stored != actual {
            return Err(Error::CorruptStorage(
                "stored Raft snapshot checksum mismatch".into(),
            ));
        }
    }
    Ok(Some(snapshot))
}

fn stored_snapshot_checksum(
    meta: &SnapshotMeta<NodeId, openraft::BasicNode>,
    data: &[u8],
) -> std::io::Result<u32> {
    serde_json::to_vec(&(meta, data))
        .map(|bytes| crc32c(&bytes))
        .map_err(invalid_data)
}

fn invalid_data(error: impl std::error::Error + Send + Sync + 'static) -> std::io::Error {
    std::io::Error::new(ErrorKind::InvalidData, error)
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f63b78_u32 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use openraft::{CommittedLeaderId, EntryPayload};

    use crate::{Filter, Metric, UpsertPoint, Value};

    use super::*;

    fn directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("focal-vector-state-{}-{nonce}", std::process::id()))
    }

    fn config() -> CollectionConfig {
        CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        }
    }
    fn entry(index: u64, request_id: u64) -> Entry<FocalRaftConfig> {
        upsert_entry(index, request_id, "p", vec![1.0, 0.0])
    }

    fn upsert_entry(
        index: u64,
        request_id: u64,
        id: &str,
        vector: Vec<f32>,
    ) -> Entry<FocalRaftConfig> {
        Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), index),
            payload: EntryPayload::Normal(ShardCommand::Upsert {
                client_id: "client".into(),
                request_id,
                points: vec![UpsertPoint {
                    id: id.into(),
                    vector,
                    metadata: BTreeMap::new(),
                }],
            }),
        }
    }

    fn delete_entry(
        index: u64,
        request_id: u64,
        ids: impl IntoIterator<Item = &'static str>,
    ) -> Entry<FocalRaftConfig> {
        Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), index),
            payload: EntryPayload::Normal(ShardCommand::Delete {
                client_id: "client".into(),
                request_id,
                ids: ids.into_iter().map(str::to_owned).collect(),
            }),
        }
    }

    #[tokio::test]
    async fn applies_deduplicates_and_recovers() {
        let directory = directory();
        let mut state = DurableShardStateMachine::open(&directory, config()).unwrap();
        let responses = state.apply([entry(1, 7), entry(2, 7)]).await.unwrap();
        assert_eq!(
            responses,
            [
                ShardResponse::Applied { sequence: 1 },
                ShardResponse::Applied { sequence: 1 }
            ]
        );
        assert_eq!(state.len().await, 1);
        drop(state);
        let mut reopened = DurableShardStateMachine::open(&directory, config()).unwrap();
        assert_eq!(reopened.len().await, 1);
        assert_eq!(reopened.applied_state().await.unwrap().0.unwrap().index, 2);
        assert_eq!(
            reopened.search(vec![1.0, 0.0], 1, None).await.unwrap()[0].id,
            "p"
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn snapshot_round_trip_replaces_state() {
        let source = directory();
        let target = directory();
        let mut state = DurableShardStateMachine::open(&source, config()).unwrap();
        state.apply([entry(1, 1)]).await.unwrap();
        let snapshot = state.build_snapshot().await.unwrap();
        let mut restored = DurableShardStateMachine::open(&target, config()).unwrap();
        restored
            .install_snapshot(&snapshot.meta, snapshot.snapshot)
            .await
            .unwrap();
        assert_eq!(restored.len().await, 1);
        drop(state);
        drop(restored);
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(target).unwrap();
    }

    #[tokio::test]
    async fn snapshot_index_merges_updates_inserts_and_deletes_exactly() {
        let directory = directory();
        let mut state = DurableShardStateMachine::open(&directory, config()).unwrap();
        state
            .apply([
                upsert_entry(1, 1, "p", vec![1.0, 0.0]),
                upsert_entry(2, 2, "q", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();
        state.build_snapshot().await.unwrap();
        assert!(state.has_approximate_index().await);
        assert_eq!(state.pending_point_count().await, 0);

        state
            .apply([
                upsert_entry(3, 3, "p", vec![-1.0, 0.0]),
                upsert_entry(4, 4, "r", vec![2.0, 0.0]),
                delete_entry(5, 5, ["q"]),
            ])
            .await
            .unwrap();
        assert_eq!(state.pending_point_count().await, 3);
        let hits = state.search(vec![1.0, 0.0], 3, None).await.unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["r", "p"]
        );

        drop(state);
        let reopened = DurableShardStateMachine::open(&directory, config()).unwrap();
        assert!(reopened.has_approximate_index().await);
        assert_eq!(reopened.pending_point_count().await, 3);
        let hits = reopened.search(vec![1.0, 0.0], 3, None).await.unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            ["r", "p"]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn snapshot_index_matches_exact_search_on_deterministic_corpus() {
        let directory = directory();
        let mut state = DurableShardStateMachine::open(&directory, config()).unwrap();
        let entries = (0..200).map(|index| {
            let angle = (index as f32) * 0.031_415_93;
            upsert_entry(
                index + 1,
                index + 1,
                &format!("p-{index:03}"),
                vec![angle.cos(), angle.sin()],
            )
        });
        state.apply(entries).await.unwrap();

        let exact = state
            .inner
            .lock()
            .await
            .collection
            .search(vec![0.31, 0.95], 10, None)
            .unwrap();
        state.build_snapshot().await.unwrap();
        let approximate = state
            .search_with_ef(vec![0.31, 0.95], 10, None, 128)
            .await
            .unwrap();
        assert_eq!(
            approximate.iter().map(|hit| &hit.id).collect::<Vec<_>>(),
            exact.iter().map(|hit| &hit.id).collect::<Vec<_>>()
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn bulk_load_builds_index_lazily_on_first_search() {
        let directory = directory();
        let mut state = DurableShardStateMachine::open(&directory, config()).unwrap();
        let points = (0..MIN_REINDEXED_POINTS)
            .map(|index| UpsertPoint {
                id: format!("bulk-{index:03}"),
                vector: vec![index as f32, 1.0],
                metadata: BTreeMap::new(),
            })
            .collect();
        state
            .apply([Entry {
                log_id: LogId::new(CommittedLeaderId::new(1, 1), 1),
                payload: EntryPayload::Normal(ShardCommand::Upsert {
                    client_id: "bulk-client".into(),
                    request_id: 1,
                    points,
                }),
            }])
            .await
            .unwrap();
        assert!(!state.has_approximate_index().await);
        assert_eq!(state.pending_point_count().await, MIN_REINDEXED_POINTS);
        assert_eq!(
            state.search(vec![1.0, 0.0], 1, None).await.unwrap()[0].id,
            "bulk-255"
        );
        assert!(state.has_approximate_index().await);
        assert_eq!(state.pending_point_count().await, 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn snapshot_checksum_rejects_corruption_and_truncation() {
        let corrupted_directory = directory();
        let mut state = DurableShardStateMachine::open(&corrupted_directory, config()).unwrap();
        state.apply([entry(1, 1)]).await.unwrap();
        state.build_snapshot().await.unwrap();
        drop(state);

        let path = corrupted_directory.join("state.snapshot");
        let mut stored: StoredSnapshot = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        stored.data[20] ^= 0x40;
        fs::write(&path, serde_json::to_vec(&stored).unwrap()).unwrap();
        assert!(matches!(
            DurableShardStateMachine::open(&corrupted_directory, config()),
            Err(Error::CorruptStorage(message)) if message.contains("checksum")
        ));
        fs::remove_dir_all(corrupted_directory).unwrap();

        let truncated_directory = directory();
        let mut state = DurableShardStateMachine::open(&truncated_directory, config()).unwrap();
        state.apply([entry(1, 1)]).await.unwrap();
        state.build_snapshot().await.unwrap();
        drop(state);
        let path = truncated_directory.join("state.snapshot");
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        assert!(matches!(
            DurableShardStateMachine::open(&truncated_directory, config()),
            Err(Error::CorruptStorage(_))
        ));
        fs::remove_dir_all(truncated_directory).unwrap();
    }

    #[tokio::test]
    async fn broad_filtered_queries_use_index_and_preserve_filter_correctness() {
        let directory = directory();
        let mut state = DurableShardStateMachine::open(&directory, config()).unwrap();
        let points = (0..1_024)
            .map(|index| UpsertPoint {
                id: format!("filtered-{index:03}"),
                vector: vec![index as f32, 1.0],
                metadata: BTreeMap::from([(
                    "group".into(),
                    Value::Keyword(if index % 2 == 0 { "even" } else { "odd" }.into()),
                )]),
            })
            .collect();
        state
            .apply([Entry {
                log_id: LogId::new(CommittedLeaderId::new(1, 1), 1),
                payload: EntryPayload::Normal(ShardCommand::Upsert {
                    client_id: "filter-client".into(),
                    request_id: 1,
                    points,
                }),
            }])
            .await
            .unwrap();
        let filter = Filter::Eq {
            field: "group".into(),
            value: Value::Keyword("even".into()),
        };
        let hits = state
            .search_with_ef(vec![1.0, 0.0], 10, Some(&filter), 128)
            .await
            .unwrap();
        assert_eq!(hits.len(), 10);
        assert!(
            hits.iter()
                .all(|hit| { hit.metadata.get("group") == Some(&Value::Keyword("even".into())) })
        );
        assert_eq!(hits[0].id, "filtered-1022");
        fs::remove_dir_all(directory).unwrap();
    }
}
