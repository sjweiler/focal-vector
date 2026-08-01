use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use focal_vector::{CollectionConfig, Durability, Metric, PersistentCollection, UpsertPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point_count = argument(1, 100_000);
    let dimension = argument(2, 128);
    let delta_count = argument(3, point_count.div_ceil(100));
    if point_count == 0 || dimension == 0 || delta_count == 0 || delta_count > point_count {
        return Err(
            "counts and dimension must be positive; delta count cannot exceed points".into(),
        );
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = env::temp_dir().join(format!(
        "focal-vector-persistence-bench-{}-{nonce}",
        std::process::id()
    ));
    let result = run(&directory, point_count, dimension, delta_count);
    let _ = fs::remove_dir_all(&directory);
    result
}

fn run(
    directory: &std::path::Path,
    point_count: usize,
    dimension: usize,
    delta_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = CollectionConfig {
        dimension,
        metric: Metric::Cosine,
    };
    let mut random = Generator::new(0x5eed_f0ca_dead_beef);
    let points: Vec<UpsertPoint> = (0..point_count)
        .map(|index| UpsertPoint {
            id: format!("point-{index:09}"),
            vector: (0..dimension).map(|_| random.value()).collect(),
            metadata: BTreeMap::new(),
        })
        .collect();
    let mut collection = PersistentCollection::open(directory, config, Durability::Sync)?;
    collection.upsert(points)?;
    let started = Instant::now();
    collection.flush()?;
    let full_elapsed = started.elapsed();
    let full_size = segment_size(directory, 1)?;
    let full_stats = collection.index_stats();

    let changes: Vec<UpsertPoint> = (0..delta_count)
        .map(|index| UpsertPoint {
            id: format!("point-{index:09}"),
            vector: (0..dimension).map(|_| random.value()).collect(),
            metadata: BTreeMap::new(),
        })
        .collect();
    collection.upsert(changes)?;
    let started = Instant::now();
    collection.flush()?;
    let delta_elapsed = started.elapsed();
    let delta_size = segment_size(directory, 2)?;
    let delta_stats = collection.index_stats();

    println!(
        "points={point_count} dimensions={dimension} delta_points={delta_count} full_flush_ms={:.2} delta_flush_ms={:.2} speedup={:.2}x",
        full_elapsed.as_secs_f64() * 1_000.0,
        delta_elapsed.as_secs_f64() * 1_000.0,
        full_elapsed.as_secs_f64() / delta_elapsed.as_secs_f64()
    );
    println!(
        "full_segment_bytes={full_size} delta_segment_bytes={delta_size} size_reduction={:.2}x mapped_points={} owned_points={} owned_vector_bytes={} index_segments={}",
        full_size as f64 / delta_size as f64,
        delta_stats.mapped_points,
        delta_stats.owned_points,
        delta_stats.owned_vector_bytes,
        delta_stats.segments
    );
    assert_eq!(full_stats.mapped_points, point_count);
    assert_eq!(delta_stats.mapped_points, point_count);
    assert_eq!(delta_stats.owned_vector_bytes, 0);
    Ok(())
}

fn segment_size(directory: &std::path::Path, sequence: u64) -> std::io::Result<u64> {
    fs::metadata(directory.join(format!("segment-{sequence:020}.fvs"))).map(|value| value.len())
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
