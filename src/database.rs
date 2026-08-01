use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::Duration;

use crate::{
    BackgroundFlusher, CollectionConfig, Durability, Error, PersistentCollection, Result,
    SharedCollection,
};

#[derive(Debug, Clone, Copy)]
pub struct DatabaseConfig {
    pub durability: Durability,
    pub flush_interval: Duration,
    pub dirty_point_threshold: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            durability: Durability::Sync,
            flush_interval: Duration::from_secs(1),
            dirty_point_threshold: 10_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSummary {
    pub name: String,
    pub config: CollectionConfig,
    pub points: usize,
    pub latest_sequence: u64,
    pub pending_points: usize,
}

#[derive(Debug)]
struct ManagedCollection {
    collection: SharedCollection,
    _flusher: BackgroundFlusher,
}

#[derive(Debug)]
pub struct Database {
    root: PathBuf,
    config: DatabaseConfig,
    collections: RwLock<HashMap<String, ManagedCollection>>,
}

impl Database {
    pub fn open(root: impl AsRef<Path>, config: DatabaseConfig) -> Result<Self> {
        validate_database_config(config)?;
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join(".backups"))?;
        let database = Self {
            root,
            config,
            collections: RwLock::new(HashMap::new()),
        };
        database.discover()?;
        Ok(database)
    }

    pub fn create_collection(
        &self,
        name: &str,
        collection_config: CollectionConfig,
    ) -> Result<SharedCollection> {
        validate_collection_name(name)?;
        let mut collections = self.write_collections()?;
        if collections.contains_key(name) {
            return Err(Error::AlreadyExists(format!("collection {name}")));
        }
        let managed = self.open_managed(name, collection_config)?;
        let collection = managed.collection.clone();
        collections.insert(name.to_owned(), managed);
        Ok(collection)
    }

    pub fn collection(&self, name: &str) -> Result<SharedCollection> {
        validate_collection_name(name)?;
        self.read_collections()?
            .get(name)
            .map(|managed| managed.collection.clone())
            .ok_or_else(|| Error::NotFound(format!("collection {name}")))
    }

    pub fn list_collections(&self) -> Result<Vec<CollectionSummary>> {
        let collections = self.read_collections()?;
        let mut summaries = Vec::with_capacity(collections.len());
        for (name, managed) in collections.iter() {
            summaries.push(CollectionSummary {
                name: name.clone(),
                config: managed.collection.config()?,
                points: managed.collection.len()?,
                latest_sequence: managed.collection.latest_sequence()?,
                pending_points: managed.collection.pending_point_count()?,
            });
        }
        summaries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        Ok(summaries)
    }

    pub fn backup_collection(&self, collection_name: &str, backup_name: &str) -> Result<u64> {
        validate_collection_name(backup_name)?;
        let collection = self.collection(collection_name)?;
        collection.backup_to(self.root.join(".backups").join(backup_name))
    }

    pub fn restore_collection(
        &self,
        backup_name: &str,
        collection_name: &str,
    ) -> Result<SharedCollection> {
        validate_collection_name(backup_name)?;
        validate_collection_name(collection_name)?;
        let mut collections = self.write_collections()?;
        if collections.contains_key(collection_name) {
            return Err(Error::AlreadyExists(format!(
                "collection {collection_name}"
            )));
        }
        let source = self.root.join(".backups").join(backup_name);
        if !source.join("collection.meta").is_file() {
            return Err(Error::NotFound(format!("backup {backup_name}")));
        }
        let destination = self.root.join(collection_name);
        let config = PersistentCollection::restore_backup(&source, &destination)?;
        match self.open_managed(collection_name, config) {
            Ok(managed) => {
                let collection = managed.collection.clone();
                collections.insert(collection_name.to_owned(), managed);
                Ok(collection)
            }
            Err(error) => {
                let _ = fs::remove_dir_all(destination);
                Err(error)
            }
        }
    }

    pub fn list_backups(&self) -> Result<Vec<String>> {
        let mut backups = Vec::new();
        for entry in fs::read_dir(self.root.join(".backups"))? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() || !entry.path().join("collection.meta").is_file() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str()
                && validate_collection_name(name).is_ok()
            {
                backups.push(name.to_owned());
            }
        }
        backups.sort_unstable();
        Ok(backups)
    }

    pub fn is_ready(&self) -> bool {
        self.collections.read().is_ok()
    }

    fn discover(&self) -> Result<()> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if validate_collection_name(&name).is_ok()
                && entry.path().join("collection.meta").is_file()
            {
                names.push(name);
            }
        }
        names.sort_unstable();
        let mut collections = self.write_collections()?;
        for name in names {
            let config = PersistentCollection::persisted_config(self.root.join(&name))?;
            let managed = self.open_managed(&name, config)?;
            collections.insert(name, managed);
        }
        Ok(())
    }

    fn open_managed(&self, name: &str, config: CollectionConfig) -> Result<ManagedCollection> {
        let collection =
            SharedCollection::open(self.root.join(name), config, self.config.durability)?;
        let flusher = collection.start_background_flush(
            self.config.flush_interval,
            self.config.dirty_point_threshold,
        )?;
        Ok(ManagedCollection {
            collection,
            _flusher: flusher,
        })
    }

    fn read_collections(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, ManagedCollection>>> {
        self.collections
            .read()
            .map_err(|_| Error::Concurrency("database catalog read lock is poisoned".into()))
    }

    fn write_collections(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, ManagedCollection>>> {
        self.collections
            .write()
            .map_err(|_| Error::Concurrency("database catalog write lock is poisoned".into()))
    }
}

fn validate_database_config(config: DatabaseConfig) -> Result<()> {
    if config.flush_interval.is_zero() || config.dirty_point_threshold == 0 {
        return Err(Error::InvalidConfig(
            "flush interval and dirty point threshold must be positive",
        ));
    }
    Ok(())
}

fn validate_collection_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err(Error::InvalidQuery(
            "collection names must be 1-64 ASCII letters, digits, '-' or '_'",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::Metric;

    use super::*;

    fn test_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-database-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn creates_lists_and_rediscovers_collections() {
        let root = test_directory();
        let config = DatabaseConfig {
            dirty_point_threshold: 100,
            ..DatabaseConfig::default()
        };
        {
            let database = Database::open(&root, config).unwrap();
            database
                .create_collection(
                    "articles",
                    CollectionConfig {
                        dimension: 3,
                        metric: Metric::Cosine,
                    },
                )
                .unwrap();
            assert_eq!(database.list_collections().unwrap()[0].name, "articles");
            assert!(matches!(
                database.create_collection(
                    "articles",
                    CollectionConfig {
                        dimension: 3,
                        metric: Metric::Cosine,
                    }
                ),
                Err(Error::AlreadyExists(_))
            ));
        }
        let reopened = Database::open(&root, config).unwrap();
        assert_eq!(
            reopened
                .collection("articles")
                .unwrap()
                .config()
                .unwrap()
                .dimension,
            3
        );
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_path_like_collection_names() {
        let root = test_directory();
        let database = Database::open(&root, DatabaseConfig::default()).unwrap();
        assert!(database.collection("../escape").is_err());
        drop(database);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn backs_up_and_restores_a_collection() {
        let root = test_directory();
        let database = Database::open(&root, DatabaseConfig::default()).unwrap();
        let original = database
            .create_collection(
                "original",
                CollectionConfig {
                    dimension: 2,
                    metric: Metric::Cosine,
                },
            )
            .unwrap();
        original
            .upsert(vec![crate::UpsertPoint {
                id: "doc".into(),
                vector: vec![1.0, 0.0],
                metadata: std::collections::BTreeMap::new(),
            }])
            .unwrap();

        assert_eq!(
            database.backup_collection("original", "backup-1").unwrap(),
            1
        );
        assert_eq!(database.list_backups().unwrap(), ["backup-1"]);
        original.delete(vec!["doc".into()]).unwrap();

        let restored = database.restore_collection("backup-1", "restored").unwrap();
        assert_eq!(restored.latest_sequence().unwrap(), 1);
        assert_eq!(
            restored.search(vec![1.0, 0.0], 1, None).unwrap()[0].id,
            "doc"
        );
        drop(original);
        drop(restored);
        drop(database);
        fs::remove_dir_all(root).unwrap();
    }
}
