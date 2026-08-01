use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Keyword(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub(crate) fn matches(&self, metadata: &BTreeMap<String, Value>) -> bool {
        match self {
            Self::MatchAll => true,
            Self::Eq { field, value } => metadata.get(field) == Some(value),
            Self::Range { field, gte, lt } => metadata
                .get(field)
                .and_then(numeric_value)
                .is_some_and(|value| {
                    gte.is_none_or(|lower| value >= lower) && lt.is_none_or(|upper| value < upper)
                }),
            Self::And(filters) => filters.iter().all(|filter| filter.matches(metadata)),
            Self::Or(filters) => filters.iter().any(|filter| filter.matches(metadata)),
            Self::Not(filter) => !filter.matches(metadata),
        }
    }
}

fn numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(value) => Some(*value as f64),
        Value::Float(value) if value.is_finite() => Some(*value),
        _ => None,
    }
}
