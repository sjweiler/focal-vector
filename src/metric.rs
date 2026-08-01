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
            Self::Cosine | Self::DotProduct => dot_product(query, candidate),
            Self::Euclidean => -squared_distance(query, candidate),
        }
    }
}

// Multiple accumulators break the dependency chain in a scalar reduction and
// give LLVM a shape it can vectorize without changing the public score type.
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0_f32; 4];
    let mut left_chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);
    for (left, right) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        sums[0] += left[0] * right[0];
        sums[1] += left[1] * right[1];
        sums[2] += left[2] * right[2];
        sums[3] += left[3] * right[3];
    }
    sums.into_iter().sum::<f32>()
        + left_chunks
            .remainder()
            .iter()
            .zip(right_chunks.remainder())
            .map(|(left, right)| left * right)
            .sum::<f32>()
}

fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0_f32; 4];
    let mut left_chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);
    for (left, right) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        let delta0 = left[0] - right[0];
        let delta1 = left[1] - right[1];
        let delta2 = left[2] - right[2];
        let delta3 = left[3] - right[3];
        sums[0] += delta0 * delta0;
        sums[1] += delta1 * delta1;
        sums[2] += delta2 * delta2;
        sums[3] += delta3 * delta3;
    }
    sums.into_iter().sum::<f32>()
        + left_chunks
            .remainder()
            .iter()
            .zip(right_chunks.remainder())
            .map(|(left, right)| {
                let delta = left - right;
                delta * delta
            })
            .sum::<f32>()
}
