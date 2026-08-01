use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;

use crate::{Filter, Point, Value};

#[derive(Debug, Clone)]
pub(crate) struct MetadataIndex {
    all_ids: HashSet<String>,
    equality: HashMap<String, HashMap<ScalarKey, HashSet<String>>>,
    numeric: HashMap<String, BTreeMap<OrderedF64, HashSet<String>>>,
}

impl MetadataIndex {
    pub(crate) fn build<'a>(points: impl IntoIterator<Item = &'a Point>) -> Self {
        let mut index = Self {
            all_ids: HashSet::new(),
            equality: HashMap::new(),
            numeric: HashMap::new(),
        };
        for point in points {
            index.all_ids.insert(point.id.clone());
            for (field, value) in &point.metadata {
                if let Some(key) = ScalarKey::from_value(value) {
                    index
                        .equality
                        .entry(field.clone())
                        .or_default()
                        .entry(key)
                        .or_default()
                        .insert(point.id.clone());
                }
                if let Some(value) = numeric_value(value) {
                    index
                        .numeric
                        .entry(field.clone())
                        .or_default()
                        .entry(OrderedF64::new(value))
                        .or_default()
                        .insert(point.id.clone());
                }
            }
        }
        index
    }

    pub(crate) fn candidates(&self, filter: &Filter) -> HashSet<String> {
        match filter {
            Filter::MatchAll => self.all_ids.clone(),
            Filter::Eq { field, value } => ScalarKey::from_value(value)
                .and_then(|key| self.equality.get(field)?.get(&key))
                .cloned()
                .unwrap_or_default(),
            Filter::Range { field, gte, lt } => self.range(field, *gte, *lt),
            Filter::And(filters) => {
                let mut filters = filters.iter();
                let Some(first) = filters.next() else {
                    return self.all_ids.clone();
                };
                let mut candidates = self.candidates(first);
                for filter in filters {
                    let next = self.candidates(filter);
                    candidates.retain(|id| next.contains(id));
                    if candidates.is_empty() {
                        break;
                    }
                }
                candidates
            }
            Filter::Or(filters) => {
                let mut candidates = HashSet::new();
                for filter in filters {
                    candidates.extend(self.candidates(filter));
                }
                candidates
            }
            Filter::Not(filter) => {
                let excluded = self.candidates(filter);
                self.all_ids.difference(&excluded).cloned().collect()
            }
        }
    }

    fn range(&self, field: &str, gte: Option<f64>, lt: Option<f64>) -> HashSet<String> {
        if gte.is_some_and(f64::is_nan) || lt.is_some_and(f64::is_nan) {
            return HashSet::new();
        }
        if gte.zip(lt).is_some_and(|(lower, upper)| lower >= upper) {
            return HashSet::new();
        }
        let Some(values) = self.numeric.get(field) else {
            return HashSet::new();
        };
        let lower = gte
            .map(|value| Bound::Included(OrderedF64::new(value)))
            .unwrap_or(Bound::Unbounded);
        let upper = lt
            .map(|value| Bound::Excluded(OrderedF64::new(value)))
            .unwrap_or(Bound::Unbounded);
        values
            .range((lower, upper))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ScalarKey {
    Keyword(String),
    Integer(i64),
    Float(u64),
    Boolean(bool),
}

impl ScalarKey {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Keyword(value) => Some(Self::Keyword(value.clone())),
            Value::Integer(value) => Some(Self::Integer(*value)),
            Value::Float(value) if !value.is_nan() => {
                let canonical = if *value == 0.0 { 0.0 } else { *value };
                Some(Self::Float(canonical.to_bits()))
            }
            Value::Float(_) => None,
            Value::Boolean(value) => Some(Self::Boolean(*value)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl OrderedF64 {
    fn new(value: f64) -> Self {
        Self(if value == 0.0 { 0.0 } else { value })
    }
}

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(value) => Some(*value as f64),
        Value::Float(value) if value.is_finite() => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: &str, tenant: &str, price: i64, enabled: bool) -> Point {
        Point {
            id: id.into(),
            vector: vec![1.0],
            metadata: BTreeMap::from([
                ("tenant".into(), Value::Keyword(tenant.into())),
                ("price".into(), Value::Integer(price)),
                ("enabled".into(), Value::Boolean(enabled)),
            ]),
            sequence: 1,
        }
    }

    #[test]
    fn combines_equality_range_and_boolean_expressions() {
        let points = [
            point("a", "one", 5, true),
            point("b", "one", 15, false),
            point("c", "two", 25, true),
        ];
        let index = MetadataIndex::build(&points);
        let filter = Filter::And(vec![
            Filter::Eq {
                field: "tenant".into(),
                value: Value::Keyword("one".into()),
            },
            Filter::Range {
                field: "price".into(),
                gte: Some(10.0),
                lt: Some(20.0),
            },
            Filter::Not(Box::new(Filter::Eq {
                field: "enabled".into(),
                value: Value::Boolean(true),
            })),
        ]);
        assert_eq!(index.candidates(&filter), HashSet::from(["b".into()]));
        assert!(
            index
                .candidates(&Filter::Range {
                    field: "price".into(),
                    gte: Some(20.0),
                    lt: Some(10.0),
                })
                .is_empty()
        );
    }
}
