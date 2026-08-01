use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{
    CollectionConfig, Durability, Error, Filter, PersistentCollection, Result, SearchHit,
    UpsertPoint,
};

#[derive(Debug, Clone)]
pub struct SharedCollection {
    inner: Arc<RwLock<PersistentCollection>>,
}

impl SharedCollection {
    pub fn open(
        directory: impl AsRef<std::path::Path>,
        config: CollectionConfig,
        durability: Durability,
    ) -> Result<Self> {
        Ok(Self::new(PersistentCollection::open(
            directory, config, durability,
        )?))
    }

    pub fn new(collection: PersistentCollection) -> Self {
        Self {
            inner: Arc::new(RwLock::new(collection)),
        }
    }

    pub fn upsert(&self, points: Vec<UpsertPoint>) -> Result<u64> {
        self.write()?.upsert(points)
    }

    pub fn delete(&self, ids: Vec<String>) -> Result<u64> {
        self.write()?.delete(ids)
    }

    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchHit>> {
        self.read()?.search(query, k, filter)
    }

    pub fn search_with_ef(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
        ef_search: usize,
    ) -> Result<Vec<SearchHit>> {
        self.read()?.search_with_ef(query, k, filter, ef_search)
    }

    pub fn flush(&self) -> Result<u64> {
        let snapshot = self.read()?.flush_snapshot();
        let prepared = PersistentCollection::build_flush(snapshot)?;
        self.write()?.publish_flush(prepared)
    }

    pub fn latest_sequence(&self) -> Result<u64> {
        Ok(self.read()?.latest_sequence())
    }

    pub fn config(&self) -> Result<CollectionConfig> {
        Ok(self.read()?.config())
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.read()?.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.read()?.is_empty())
    }

    pub fn pending_point_count(&self) -> Result<usize> {
        Ok(self.read()?.pending_point_count())
    }

    pub fn has_approximate_index(&self) -> Result<bool> {
        Ok(self.read()?.has_approximate_index())
    }

    pub fn backup_to(&self, destination: impl AsRef<std::path::Path>) -> Result<u64> {
        self.flush()?;
        let collection = self.read()?;
        let sequence = collection.latest_sequence();
        collection.copy_backup_to(destination.as_ref())?;
        Ok(sequence)
    }

    pub fn start_background_flush(
        &self,
        check_interval: Duration,
        dirty_point_threshold: usize,
    ) -> Result<BackgroundFlusher> {
        if dirty_point_threshold == 0 {
            return Err(Error::InvalidConfig(
                "background flush threshold must be greater than zero",
            ));
        }
        if check_interval.is_zero() {
            return Err(Error::InvalidConfig(
                "background flush interval must be greater than zero",
            ));
        }

        let collection = self.clone();
        let (stop_sender, stop_receiver) = mpsc::channel();
        let last_error = Arc::new(Mutex::new(None));
        let thread_error = Arc::clone(&last_error);
        let join = thread::Builder::new()
            .name("focal-vector-flush".into())
            .spawn(move || {
                loop {
                    match stop_receiver.recv_timeout(check_interval) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    let result = (|| -> Result<()> {
                        if collection.pending_point_count()? < dirty_point_threshold {
                            return Ok(());
                        }
                        collection.flush()?;
                        Ok(())
                    })();
                    if let Err(error) = result {
                        if let Ok(mut slot) = thread_error.lock() {
                            *slot = Some(error);
                        }
                        break;
                    }
                }
            })
            .map_err(|error| Error::Concurrency(error.to_string()))?;

        Ok(BackgroundFlusher {
            stop_sender: Some(stop_sender),
            join: Some(join),
            last_error,
        })
    }

    fn read(&self) -> Result<std::sync::RwLockReadGuard<'_, PersistentCollection>> {
        self.inner
            .read()
            .map_err(|_| Error::Concurrency("collection read lock is poisoned".into()))
    }

    fn write(&self) -> Result<std::sync::RwLockWriteGuard<'_, PersistentCollection>> {
        self.inner
            .write()
            .map_err(|_| Error::Concurrency("collection write lock is poisoned".into()))
    }
}

#[derive(Debug)]
pub struct BackgroundFlusher {
    stop_sender: Option<mpsc::Sender<()>>,
    join: Option<JoinHandle<()>>,
    last_error: Arc<Mutex<Option<Error>>>,
}

impl BackgroundFlusher {
    pub fn last_error(&self) -> Option<Error> {
        self.last_error.lock().ok().and_then(|error| error.clone())
    }

    pub fn stop(mut self) -> Result<()> {
        self.stop_and_join();
        match self.last_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn stop_and_join(&mut self) {
        if let Some(sender) = self.stop_sender.take() {
            let _ = sender.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for BackgroundFlusher {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use crate::{Metric, UpsertPoint};

    use super::*;

    fn test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-shared-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn point(id: &str, vector: [f32; 2]) -> UpsertPoint {
        UpsertPoint {
            id: id.into(),
            vector: vector.into(),
            metadata: BTreeMap::new(),
        }
    }

    fn config() -> CollectionConfig {
        CollectionConfig {
            dimension: 2,
            metric: Metric::Cosine,
        }
    }

    #[test]
    fn supports_concurrent_readers_and_writer() {
        let directory = test_directory("read-write");
        let collection = SharedCollection::open(&directory, config(), Durability::Sync).unwrap();
        collection.upsert(vec![point("seed", [1.0, 0.0])]).unwrap();

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let collection = collection.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        assert!(
                            !collection
                                .search(vec![1.0, 0.0], 5, None)
                                .unwrap()
                                .is_empty()
                        );
                    }
                })
            })
            .collect();
        for index in 0..20 {
            collection
                .upsert(vec![point(
                    &format!("p-{index}"),
                    [1.0, index as f32 + 1.0],
                )])
                .unwrap();
        }
        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(collection.latest_sequence().unwrap(), 21);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn background_flusher_checkpoints_at_threshold() {
        let directory = test_directory("background-flush");
        let collection = SharedCollection::open(&directory, config(), Durability::Sync).unwrap();
        let flusher = collection
            .start_background_flush(Duration::from_millis(5), 2)
            .unwrap();
        collection
            .upsert(vec![point("a", [1.0, 0.0]), point("b", [0.0, 1.0])])
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while collection.pending_point_count().unwrap() != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(collection.pending_point_count().unwrap(), 0);
        assert!(collection.has_approximate_index().unwrap());
        flusher.stop().unwrap();
        drop(collection);

        let reopened = SharedCollection::open(&directory, config(), Durability::Sync).unwrap();
        assert_eq!(reopened.search(vec![1.0, 0.0], 2, None).unwrap().len(), 2);
        fs::remove_dir_all(directory).unwrap();
    }
}
