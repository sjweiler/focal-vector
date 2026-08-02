use std::env;

use crate::{Error, Result};

/// Controls the optional CUDA exact-search cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaSearchMode {
    Disabled,
    /// Use CUDA when it is available and the collection is large enough.
    Preferred,
    /// Require CUDA to initialize when the collection is opened.
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSearchConfig {
    pub mode: CudaSearchMode,
    pub device: usize,
    /// `Preferred` mode leaves smaller collections on CPU/HNSW.
    pub min_vectors: usize,
}

impl Default for CudaSearchConfig {
    fn default() -> Self {
        Self {
            mode: CudaSearchMode::Disabled,
            device: 0,
            min_vectors: 10_000,
        }
    }
}

impl CudaSearchConfig {
    pub(crate) fn from_env() -> Result<Self> {
        let Some(mode) = env::var_os("FOCAL_CUDA") else {
            return Ok(Self::default());
        };
        let mode = mode
            .into_string()
            .map_err(|_| Error::InvalidConfiguration("FOCAL_CUDA is not UTF-8".into()))?;
        let mode = match mode.to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "disabled" => CudaSearchMode::Disabled,
            "1" | "on" | "true" | "auto" | "preferred" => CudaSearchMode::Preferred,
            "required" => CudaSearchMode::Required,
            _ => {
                return Err(Error::InvalidConfiguration(
                    "FOCAL_CUDA must be off, auto, or required".into(),
                ));
            }
        };
        let device = parse_usize_env("FOCAL_CUDA_DEVICE", 0)?;
        let min_vectors = parse_usize_env("FOCAL_CUDA_MIN_VECTORS", 10_000)?;
        Ok(Self {
            mode,
            device,
            min_vectors,
        })
    }
}

fn parse_usize_env(name: &str, default: usize) -> Result<usize> {
    let Some(value) = env::var_os(name) else {
        return Ok(default);
    };
    value
        .into_string()
        .map_err(|_| Error::InvalidConfiguration(format!("{name} is not UTF-8")))?
        .parse()
        .map_err(|_| Error::InvalidConfiguration(format!("{name} must be a non-negative integer")))
}

#[cfg(feature = "cuda")]
mod enabled {
    use std::collections::HashSet;
    use std::fmt;
    use std::sync::Mutex;

    use cudarc::cublas::{CudaBlas, Gemv, GemvConfig, sys};
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

    use crate::collection::StoredPoint;
    use crate::{CollectionConfig, Error, Metric, Result};

    #[derive(Debug)]
    pub(crate) struct CudaHit {
        pub id: String,
        pub score: f32,
    }

    struct CudaState {
        stream: std::sync::Arc<CudaStream>,
        blas: CudaBlas,
        vectors: CudaSlice<f32>,
    }

    pub(crate) struct CudaIndex {
        config: CollectionConfig,
        ids: Vec<String>,
        squared_norms: Vec<f32>,
        state: Mutex<CudaState>,
    }

    impl fmt::Debug for CudaIndex {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("CudaIndex")
                .field("dimension", &self.config.dimension)
                .field("metric", &self.config.metric)
                .field("vectors", &self.ids.len())
                .finish_non_exhaustive()
        }
    }

    impl CudaIndex {
        pub(crate) fn probe(device: usize) -> Result<()> {
            let context = CudaContext::new(device)
                .map_err(|error| cuda_error("initialize CUDA device", error))?;
            CudaBlas::new(context.default_stream())
                .map_err(|error| cuda_error("initialize cuBLAS", error))?;
            Ok(())
        }

        pub(crate) fn build(
            config: CollectionConfig,
            points: &[StoredPoint],
            device: usize,
        ) -> Result<Self> {
            if points.is_empty() {
                return Err(Error::InvalidConfiguration(
                    "cannot create a CUDA index for an empty collection".into(),
                ));
            }
            let context = CudaContext::new(device)
                .map_err(|error| cuda_error("initialize CUDA device", error))?;
            let stream = context.default_stream();
            let blas = CudaBlas::new(stream.clone())
                .map_err(|error| cuda_error("initialize cuBLAS", error))?;

            let components = points.len().checked_mul(config.dimension).ok_or_else(|| {
                Error::ResourceExhausted("CUDA vector matrix is too large".into())
            })?;
            let mut host_vectors = Vec::with_capacity(components);
            let mut ids = Vec::with_capacity(points.len());
            let mut squared_norms = Vec::with_capacity(points.len());
            for point in points {
                ids.push(point.id.clone());
                host_vectors.extend_from_slice(point.vector.as_slice());
                squared_norms.push(
                    point
                        .vector
                        .iter()
                        .map(|value| (*value as f64) * (*value as f64))
                        .sum::<f64>() as f32,
                );
            }
            let vectors = stream
                .clone_htod(&host_vectors)
                .map_err(|error| cuda_error("upload CUDA vector snapshot", error))?;
            Ok(Self {
                config,
                ids,
                squared_norms,
                state: Mutex::new(CudaState {
                    stream,
                    blas,
                    vectors,
                }),
            })
        }

        pub(crate) fn search(
            &self,
            query: &[f32],
            k: usize,
            excluded_ids: &HashSet<String>,
        ) -> Result<Vec<CudaHit>> {
            let dimension = i32::try_from(self.config.dimension)
                .map_err(|_| Error::ResourceExhausted("CUDA dimension exceeds i32".into()))?;
            let rows = i32::try_from(self.ids.len())
                .map_err(|_| Error::ResourceExhausted("CUDA row count exceeds i32".into()))?;
            let state = self
                .state
                .lock()
                .map_err(|_| Error::Concurrency("CUDA search lock is poisoned".into()))?;
            let query_device = state
                .stream
                .clone_htod(query)
                .map_err(|error| cuda_error("upload CUDA query", error))?;
            let mut scores_device = state
                .stream
                .alloc_zeros::<f32>(self.ids.len())
                .map_err(|error| cuda_error("allocate CUDA score buffer", error))?;
            // Host vectors are row-major. cuBLAS sees the same bytes as a
            // column-major (dimension x rows) matrix and transposes it.
            unsafe {
                state.blas.gemv(
                    GemvConfig {
                        trans: sys::cublasOperation_t::CUBLAS_OP_T,
                        m: dimension,
                        n: rows,
                        alpha: 1.0,
                        lda: dimension,
                        incx: 1,
                        beta: 0.0,
                        incy: 1,
                    },
                    &state.vectors,
                    &query_device,
                    &mut scores_device,
                )
            }
            .map_err(|error| cuda_error("score CUDA vectors", error))?;
            let dot_products = state
                .stream
                .clone_dtoh(&scores_device)
                .map_err(|error| cuda_error("download CUDA scores", error))?;
            drop(state);

            let query_norm = if self.config.metric == Metric::Euclidean {
                query
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>() as f32
            } else {
                0.0
            };
            let hits: Vec<CudaHit> = self
                .ids
                .iter()
                .zip(dot_products)
                .enumerate()
                .filter(|(_, (id, _))| !excluded_ids.contains(id.as_str()))
                .map(|(ordinal, (id, dot_product))| {
                    let score = match self.config.metric {
                        Metric::Cosine | Metric::DotProduct => dot_product,
                        Metric::Euclidean => {
                            -(query_norm + self.squared_norms[ordinal] - 2.0 * dot_product)
                        }
                    };
                    CudaHit {
                        id: id.clone(),
                        score,
                    }
                })
                .collect();
            Ok(select_top_k(hits, k))
        }

        pub(crate) fn len(&self) -> usize {
            self.ids.len()
        }
    }

    fn cuda_hit_order(left: &CudaHit, right: &CudaHit) -> std::cmp::Ordering {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    }

    fn select_top_k(mut hits: Vec<CudaHit>, k: usize) -> Vec<CudaHit> {
        if hits.len() > k {
            hits.select_nth_unstable_by(k, cuda_hit_order);
            hits.truncate(k);
        }
        hits.sort_unstable_by(cuda_hit_order);
        hits
    }

    fn cuda_error(context: &str, error: impl fmt::Display) -> Error {
        Error::ResourceExhausted(format!("{context}: {error}"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn top_k_is_score_first_and_id_stable() {
            let hits = vec![
                CudaHit {
                    id: "z".into(),
                    score: 2.0,
                },
                CudaHit {
                    id: "low".into(),
                    score: 1.0,
                },
                CudaHit {
                    id: "a".into(),
                    score: 2.0,
                },
            ];
            let hits = select_top_k(hits, 2);
            assert_eq!(
                hits.into_iter().map(|hit| hit.id).collect::<Vec<_>>(),
                ["a", "z"]
            );
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use enabled::CudaIndex;
