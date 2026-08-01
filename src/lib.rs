//! Core types and the exact-search reference engine for Focal Vector.
//!
//! The exact engine is intentionally small and dependency-free. It defines the
//! correctness semantics that approximate indexes must preserve.

mod collection;
mod concurrent;
mod database;
mod distributed;
mod error;
mod filter;
mod hnsw;
mod metadata_index;
mod metric;
mod persistence;
mod raft_node;
mod raft_state_machine;
mod raft_storage;
mod server;
mod sharding;

pub use collection::{Collection, CollectionConfig, Point, SearchHit, UpsertPoint};
pub use concurrent::{BackgroundFlusher, SharedCollection};
pub use database::{CollectionSummary, Database, DatabaseConfig};
pub use distributed::{DistributedCollection, ReplicaSet};
pub use error::{Error, Result};
pub use filter::{Filter, Value};
pub use hnsw::{HnswConfig, HnswHit, HnswIndex};
pub use metric::Metric;
pub use persistence::{Durability, PersistentCollection};
pub use raft_node::{FocalRaft, HttpRaftNetwork, RaftNode, raft_router};
pub use raft_state_machine::DurableShardStateMachine;
pub use raft_storage::{DurableRaftLog, FocalRaftConfig, NodeId, ShardCommand, ShardResponse};
pub use server::{ServerConfig, router};
pub use sharding::{ShardCommit, ShardedCollection, shard_index};
