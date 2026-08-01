use std::collections::{BTreeMap, HashSet};
use std::env;
use std::hint::black_box;
use std::time::Instant;

use focal_vector::{Collection, CollectionConfig, HnswConfig, HnswIndex, Metric, UpsertPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point_count = argument(1, 10_000);
    let dimension = argument(2, 128);
    let query_count = argument(3, 100);
    let ef_search = argument(4, 96);
    let m = argument(5, HnswConfig::default().m);
    let ef_construction = argument(6, HnswConfig::default().ef_construction);
    let ef_search_values = ef_search_values(ef_search)?;
    let k = 10.min(point_count);
    if point_count == 0 || dimension == 0 || query_count == 0 {
        return Err("point count, dimension, and query count must be positive".into());
    }

    let mut random = Generator::new(0x5eed_f0ca_1ace_0001);
    let points: Vec<(String, Vec<f32>)> = (0..point_count)
        .map(|index| {
            (
                format!("point-{index:09}"),
                (0..dimension).map(|_| random.value()).collect(),
            )
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..query_count)
        .map(|index| {
            let mut query = points[index % point_count].1.clone();
            for value in &mut query {
                *value += random.value() * 0.01;
            }
            query
        })
        .collect();

    let mut exact = Collection::new(CollectionConfig {
        dimension,
        metric: Metric::Cosine,
    })?;
    exact.upsert(
        points
            .iter()
            .map(|(id, vector)| UpsertPoint {
                id: id.clone(),
                vector: vector.clone(),
                metadata: BTreeMap::new(),
            })
            .collect(),
    )?;

    let build_started = Instant::now();
    let approximate = HnswIndex::build(
        dimension,
        Metric::Cosine,
        HnswConfig { m, ef_construction },
        points,
    )?;
    let build_elapsed = build_started.elapsed();

    let exact_started = Instant::now();
    let exact_results: Vec<Vec<String>> = queries
        .iter()
        .map(|query| {
            exact
                .search(query.clone(), k, None)
                .map(|hits| hits.into_iter().map(|hit| hit.id).collect())
        })
        .collect::<Result<_, _>>()?;
    let exact_elapsed = exact_started.elapsed();

    println!(
        "points={point_count} dimensions={dimension} queries={query_count} k={k} m={m} ef_construction={ef_construction}"
    );
    println!("hnsw_build_ms={:.2}", build_elapsed.as_secs_f64() * 1_000.0);
    println!(
        "exact_qps={:.1} exact_ms_per_query={:.3}",
        query_count as f64 / exact_elapsed.as_secs_f64(),
        exact_elapsed.as_secs_f64() * 1_000.0 / query_count as f64
    );
    for ef_search in ef_search_values {
        let approximate_started = Instant::now();
        let approximate_results: Vec<HashSet<String>> = queries
            .iter()
            .map(|query| {
                approximate
                    .search(query.clone(), k, ef_search.max(k))
                    .map(|hits| hits.into_iter().map(|hit| hit.id).collect())
            })
            .collect::<Result<_, _>>()?;
        let approximate_elapsed = approximate_started.elapsed();
        let matches: usize = exact_results
            .iter()
            .zip(&approximate_results)
            .map(|(expected, actual)| expected.iter().filter(|id| actual.contains(*id)).count())
            .sum();
        let recall = matches as f64 / (query_count * k) as f64;
        black_box(&approximate_results);
        println!(
            "ef_search={ef_search} hnsw_qps={:.1} hnsw_ms_per_query={:.3} recall_at_{k}={recall:.4}",
            query_count as f64 / approximate_elapsed.as_secs_f64(),
            approximate_elapsed.as_secs_f64() * 1_000.0 / query_count as f64
        );
    }
    black_box(&exact_results);
    Ok(())
}

fn ef_search_values(default: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let Some(values) = env::var_os("FOCAL_BENCH_EF_SWEEP") else {
        return Ok(vec![default]);
    };
    let values = values
        .to_str()
        .ok_or("FOCAL_BENCH_EF_SWEEP must be valid UTF-8")?;
    let parsed: Vec<usize> = values
        .split(',')
        .map(|value| value.trim().parse())
        .collect::<Result<_, _>>()?;
    if parsed.is_empty() || parsed.contains(&0) {
        return Err("FOCAL_BENCH_EF_SWEEP must contain positive integers".into());
    }
    Ok(parsed)
}

fn argument(position: usize, default: usize) -> usize {
    env::args()
        .nth(position)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("argument {position} must be a positive integer"))
        })
        .unwrap_or(default)
}

struct Generator(u64);

impl Generator {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn value(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / (1_u32 << 24) as f32;
        unit * 2.0 - 1.0
    }
}
