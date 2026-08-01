//! Small, engine-independent wire types for Focal Vector's local IPC surface.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_SOCKET_NAME: &str = "focal-vector.sock";
pub const SOCKET_ENV: &str = "FOCAL_VECTOR_SOCKET";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub protocol_version: u16,
    pub payload: T,
}

impl<T> Envelope<T> {
    pub fn current(payload: T) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Cosine,
    DotProduct,
    Euclidean,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Filter {
    MatchAll,
    Eq {
        field: String,
        value: Value,
    },
    Range {
        field: String,
        gte: Option<f64>,
        lt: Option<f64>,
    },
    And {
        filters: Vec<Filter>,
    },
    Or {
        filters: Vec<Filter>,
    },
    Not {
        filter: Box<Filter>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Hello,
    ListCollections,
    CreateCollection {
        name: String,
        dimension: usize,
        metric: Metric,
    },
    Upsert {
        collection: String,
        points: Vec<Point>,
    },
    Delete {
        collection: String,
        ids: Vec<String>,
    },
    Query {
        collection: String,
        vector: Vec<f32>,
        k: usize,
        #[serde(default)]
        filter: Option<Filter>,
        #[serde(default)]
        ef_search: Option<usize>,
    },
    Flush {
        collection: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub points: usize,
    pub latest_sequence: u64,
    pub pending_points: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub metadata: BTreeMap<String, Value>,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Hello { server_version: String },
    Collections { collections: Vec<CollectionInfo> },
    Created { collection: CollectionInfo },
    Sequence { sequence: u64 },
    Query { hits: Vec<SearchHit> },
    Error { code: String, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_round_trips_with_explicit_version() {
        let encoded = serde_json::to_vec(&Envelope::current(Request::Hello)).unwrap();
        let decoded: Envelope<Request> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
        assert_eq!(decoded.payload, Request::Hello);
    }
}
