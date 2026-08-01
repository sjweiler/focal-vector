use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::RangeBounds;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openraft::storage::{LogFlushed, LogState, RaftLogReader, RaftLogStorage};
use openraft::{
    BasicNode, Entry, LogId, OptionalSend, RaftLogId, StorageError, StorageIOError, Vote,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::UpsertPoint;

const JOURNAL_MAGIC: &[u8; 4] = b"FVRJ";
const JOURNAL_VERSION: u8 = 1;
const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;

pub type NodeId = u64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShardCommand {
    Upsert {
        client_id: String,
        request_id: u64,
        points: Vec<UpsertPoint>,
    },
    Delete {
        client_id: String,
        request_id: u64,
        ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardResponse {
    Applied { sequence: u64 },
    Rejected { message: String },
}

openraft::declare_raft_types!(
    pub FocalRaftConfig:
        D = ShardCommand,
        R = ShardResponse,
        Node = BasicNode,
);

#[derive(Debug, Clone, Serialize, Deserialize)]
enum JournalRecord {
    SaveVote(Vote<NodeId>),
    SaveCommitted(Option<LogId<NodeId>>),
    Append(Vec<Entry<FocalRaftConfig>>),
    Truncate(LogId<NodeId>),
    Purge(LogId<NodeId>),
}

#[derive(Debug, Default)]
struct JournalState {
    vote: Option<Vote<NodeId>>,
    committed: Option<LogId<NodeId>>,
    last_purged: Option<LogId<NodeId>>,
    entries: BTreeMap<u64, Entry<FocalRaftConfig>>,
}

impl JournalState {
    fn apply(&mut self, record: JournalRecord) {
        match record {
            JournalRecord::SaveVote(vote) => self.vote = Some(vote),
            JournalRecord::SaveCommitted(committed) => self.committed = committed,
            JournalRecord::Append(entries) => {
                for entry in entries {
                    self.entries.insert(entry.log_id.index, entry);
                }
            }
            JournalRecord::Truncate(log_id) => {
                self.entries.split_off(&log_id.index);
            }
            JournalRecord::Purge(log_id) => {
                self.last_purged = Some(log_id);
                self.entries = self.entries.split_off(&(log_id.index + 1));
            }
        }
    }
}

#[derive(Debug)]
struct JournalInner {
    file: File,
    state: JournalState,
}

/// A durable OpenRaft log store backed by an append-only CRC32C journal.
///
/// Every write operation, including vote changes, is serialized by one mutex
/// and reaches stable storage before its completion is reported to OpenRaft.
#[derive(Debug, Clone)]
pub struct DurableRaftLog {
    directory: Arc<PathBuf>,
    inner: Arc<Mutex<JournalInner>>,
}

impl DurableRaftLog {
    pub fn open(directory: impl AsRef<Path>) -> std::io::Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let path = directory.join("raft.journal");
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let state = recover_journal(&mut file)?;
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            directory: Arc::new(directory),
            inner: Arc::new(Mutex::new(JournalInner { file, state })),
        })
    }

    pub fn directory(&self) -> &Path {
        self.directory.as_ref()
    }
}

impl RaftLogReader<FocalRaftConfig> for DurableRaftLog {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<FocalRaftConfig>>, StorageError<NodeId>>
    where
        RB: RangeBounds<u64> + Clone + Debug + OptionalSend,
    {
        let inner = self.inner.lock().await;
        Ok(inner
            .state
            .entries
            .range(range)
            .map(|(_, entry)| entry.clone())
            .collect())
    }
}

impl RaftLogStorage<FocalRaftConfig> for DurableRaftLog {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<FocalRaftConfig>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        let last_log_id = inner
            .state
            .entries
            .last_key_value()
            .map(|(_, entry)| *entry.get_log_id())
            .or(inner.state.last_purged);
        Ok(LogState {
            last_purged_log_id: inner.state.last_purged,
            last_log_id,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.write_record(JournalRecord::SaveVote(*vote), StorageKind::Vote)
            .await
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        Ok(self.inner.lock().await.state.vote)
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        self.write_record(JournalRecord::SaveCommitted(committed), StorageKind::Logs)
            .await
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        Ok(self.inner.lock().await.state.committed)
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<FocalRaftConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<FocalRaftConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        let result = self
            .write_record(JournalRecord::Append(entries), StorageKind::Logs)
            .await;
        match result {
            Ok(()) => {
                callback.log_io_completed(Ok(()));
                Ok(())
            }
            Err(error) => {
                callback.log_io_completed(Err(std::io::Error::other(error.to_string())));
                Err(error)
            }
        }
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.write_record(JournalRecord::Truncate(log_id), StorageKind::Logs)
            .await
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.write_record(JournalRecord::Purge(log_id), StorageKind::Logs)
            .await
    }
}

impl DurableRaftLog {
    async fn write_record(
        &self,
        record: JournalRecord,
        kind: StorageKind,
    ) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        if let Err(error) = append_record(&mut inner.file, &record) {
            return Err(kind.error(error));
        }
        inner.state.apply(record);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum StorageKind {
    Vote,
    Logs,
}

impl StorageKind {
    fn error(self, error: std::io::Error) -> StorageError<NodeId> {
        match self {
            Self::Vote => StorageIOError::write_vote(&error).into(),
            Self::Logs => StorageIOError::write_logs(&error).into(),
        }
    }
}

fn append_record(file: &mut File, record: &JournalRecord) -> std::io::Result<()> {
    let payload = serde_json::to_vec(record).map_err(invalid_data)?;
    if payload.len() > MAX_RECORD_BYTES {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "Raft journal record exceeds size limit",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "record is too large"))?;
    let start = file.seek(SeekFrom::End(0))?;
    let result = (|| {
        file.write_all(JOURNAL_MAGIC)?;
        file.write_all(&[JOURNAL_VERSION])?;
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

fn recover_journal(file: &mut File) -> std::io::Result<JournalState> {
    file.seek(SeekFrom::Start(0))?;
    let mut state = JournalState::default();
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
        if &header[..4] != JOURNAL_MAGIC || header[4] != JOURNAL_VERSION {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "invalid Raft journal header",
            ));
        }
        let length = u32::from_le_bytes(header[5..9].try_into().expect("four bytes")) as usize;
        if length > MAX_RECORD_BYTES {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "Raft journal record exceeds size limit",
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
        let stored_crc = u32::from_le_bytes(frame[length..].try_into().expect("four bytes"));
        if crc32c(&frame[..length]) != stored_crc {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "Raft journal checksum mismatch",
            ));
        }
        let record = serde_json::from_slice(&frame[..length]).map_err(invalid_data)?;
        state.apply(record);
    }
    Ok(state)
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use openraft::entry::RaftEntry;
    use openraft::{CommittedLeaderId, Entry, LogId};

    use super::*;

    fn directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-raft-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn log_id(index: u64) -> LogId<NodeId> {
        LogId::new(CommittedLeaderId::new(1, 1), index)
    }

    #[tokio::test]
    async fn journal_recovers_vote_commit_logs_truncation_and_purge() {
        let directory = directory("recover");
        let mut store = DurableRaftLog::open(&directory).unwrap();
        let vote = Vote::new(3, 1);
        store.save_vote(&vote).await.unwrap();
        store.save_committed(Some(log_id(2))).await.unwrap();

        // Test the durable record primitive directly because OpenRaft owns the
        // LogFlushed callback constructor.
        store
            .write_record(
                JournalRecord::Append(
                    (0..4)
                        .map(|index| Entry::new_blank(log_id(index)))
                        .collect(),
                ),
                StorageKind::Logs,
            )
            .await
            .unwrap();
        store
            .write_record(JournalRecord::Truncate(log_id(3)), StorageKind::Logs)
            .await
            .unwrap();
        store
            .write_record(JournalRecord::Purge(log_id(0)), StorageKind::Logs)
            .await
            .unwrap();
        drop(store);

        let mut reopened = DurableRaftLog::open(&directory).unwrap();
        assert_eq!(reopened.read_vote().await.unwrap(), Some(vote));
        assert_eq!(reopened.read_committed().await.unwrap(), Some(log_id(2)));
        let state = reopened.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, Some(log_id(0)));
        assert_eq!(state.last_log_id, Some(log_id(2)));
        let entries = reopened.try_get_log_entries(0..10).await.unwrap();
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.log_id.index)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        drop(reopened);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn torn_final_record_is_discarded_but_checksum_corruption_fails_closed() {
        let directory = directory("torn");
        let store = DurableRaftLog::open(&directory).unwrap();
        {
            let mut inner = store.inner.blocking_lock();
            append_record(&mut inner.file, &JournalRecord::SaveVote(Vote::new(1, 1))).unwrap();
            inner.file.write_all(JOURNAL_MAGIC).unwrap();
            inner.file.sync_data().unwrap();
        }
        drop(store);
        let reopened = DurableRaftLog::open(&directory).unwrap();
        drop(reopened);

        let path = directory.join("raft.journal");
        let mut bytes = fs::read(&path).unwrap();
        bytes[10] ^= 0x40;
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            DurableRaftLog::open(&directory).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
