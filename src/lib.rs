//! Core types and the exact-search reference engine for Focal Vector.
//!
//! The exact engine is intentionally small and dependency-free. It defines the
//! correctness semantics that approximate indexes must preserve.

mod collection;
mod error;
mod filter;
mod hnsw;
mod metric;
mod persistence;

pub use collection::{Collection, CollectionConfig, Point, SearchHit, UpsertPoint};
pub use error::{Error, Result};
pub use filter::{Filter, Value};
pub use hnsw::{HnswConfig, HnswHit, HnswIndex};
pub use metric::Metric;
pub use persistence::{Durability, PersistentCollection};
