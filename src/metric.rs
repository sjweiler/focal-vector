use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl Metric {
    pub(crate) fn prepare(self, mut vector: Vec<f32>) -> Result<Vec<f32>> {
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidVector("components must be finite"));
        }

        if self == Self::Cosine {
            let squared_norm = vector
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>();
            if squared_norm == 0.0 || !squared_norm.is_finite() {
                return Err(Error::InvalidVector(
                    "cosine similarity requires a finite, non-zero norm",
                ));
            }
            let inverse_norm = squared_norm.sqrt().recip() as f32;
            for value in &mut vector {
                *value *= inverse_norm;
            }
        }

        Ok(vector)
    }

    /// Returns a score where larger values are always better.
    pub(crate) fn score(self, query: &[f32], candidate: &[f32]) -> f32 {
        match self {
            Self::Cosine | Self::DotProduct => query
                .iter()
                .zip(candidate)
                .map(|(left, right)| left * right)
                .sum(),
            Self::Euclidean => -query
                .iter()
                .zip(candidate)
                .map(|(left, right)| {
                    let delta = left - right;
                    delta * delta
                })
                .sum::<f32>(),
        }
    }
}
