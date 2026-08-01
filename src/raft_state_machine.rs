use std::collections::BTreeMap;
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

use crate::raft_storage::{FocalRaftConfig, NodeId, ShardCommand, ShardResponse};
use crate::{Collection, CollectionConfig, Error, Filter, Result as FocalResult, SearchHit};

const STATE_MAGIC: &[u8; 4] = b"FVRS";
const STATE_VERSION: u8 = 1;
const MAX_STATE_RECORD_BYTES: usize = 256 * 1024 * 1024;

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
}

#[derive(Debug)]
struct StateInner {
    journal: File,
    collection: Collection,
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
}

impl DurableShardStateMachine {
    pub fn open(directory: impl AsRef<Path>, config: CollectionConfig) -> FocalResult<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let mut collection = Collection::new(config)?;
        let mut last_applied = None;
        let mut membership = StoredMembership::default();
        let mut dedup = BTreeMap::new();
        let current_snapshot = read_snapshot(&directory)?;
        if let Some(stored) = &current_snapshot {
            let snapshot: StateSnapshot = serde_json::from_slice(&stored.data)
                .map_err(|error| Error::CorruptStorage(error.to_string()))?;
            if snapshot.config != config {
                return Err(Error::CorruptStorage(
                    "Raft snapshot collection configuration mismatch".into(),
                ));
            }
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
                last_applied,
                membership,
                dedup,
                current_snapshot,
            })),
        })
    }

    pub async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> FocalResult<Vec<SearchHit>> {
        self.inner.lock().await.collection.search(query, k, filter)
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
        let decoded: StateSnapshot = serde_json::from_slice(&data)
            .map_err(|error| StorageIOError::read_snapshot(Some(meta.signature()), &error))?;
        if decoded.config != self.config {
            let error = std::io::Error::new(ErrorKind::InvalidData, "snapshot config mismatch");
            return Err(StorageIOError::read_snapshot(Some(meta.signature()), &error).into());
        }
        let stored = StoredSnapshot {
            meta: meta.clone(),
            data,
        };
        write_snapshot(&self.snapshot_path(), &stored)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;

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
        let data = serde_json::to_vec(&snapshot)
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
        let stored = StoredSnapshot {
            meta: meta.clone(),
            data: data.clone(),
        };
        write_snapshot(&self.snapshot_path(), &stored)
            .map_err(|error| StorageIOError::write_snapshot(Some(meta.signature()), &error))?;
        inner.current_snapshot = Some(stored);
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
                (ShardCommand::Delete { ids, .. }, Some(sequence))
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
        record,
    )
}

fn replay_record(
    collection: &mut Collection,
    last_applied: &mut Option<LogId<NodeId>>,
    membership: &mut StoredMembership<NodeId, openraft::BasicNode>,
    dedup: &mut BTreeMap<String, BTreeMap<u64, ShardResponse>>,
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
                            let prepared = collection.prepare_upsert(points.clone(), sequence)?;
                            collection.apply_prepared_upsert(prepared, sequence);
                        }
                        ShardCommand::Delete { ids, .. } => {
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
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| Error::CorruptStorage(error.to_string()))
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

    use crate::{Metric, UpsertPoint};

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
        Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), index),
            payload: EntryPayload::Normal(ShardCommand::Upsert {
                client_id: "client".into(),
                request_id,
                points: vec![UpsertPoint {
                    id: "p".into(),
                    vector: vec![1.0, 0.0],
                    metadata: BTreeMap::new(),
                }],
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
}
