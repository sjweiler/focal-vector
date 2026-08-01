use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use focal_vector_protocol as protocol;
use serde_json::Value as JsonValue;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

use crate::{
    CollectionConfig, Database, Error, Filter, Metric, Result, ServerConfig, UpsertPoint, Value,
};

/// Serve the private, versioned local IPC interface used by FocalDesk services.
pub async fn serve_local(
    socket: impl AsRef<Path>,
    database: Arc<Database>,
    config: ServerConfig,
) -> Result<()> {
    validate_limits(&config)?;
    let socket = socket.as_ref().to_path_buf();
    prepare_socket_path(&socket)?;
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let _cleanup = SocketCleanup(socket.clone());
    let state = Arc::new(LocalState {
        database,
        max_body_bytes: config.max_body_bytes,
        max_batch_points: config.max_batch_points,
        max_k: config.max_k,
        max_dimension: config.max_dimension,
        max_ef_search: config.max_ef_search,
        admission: Arc::new(Semaphore::new(config.max_concurrent_operations)),
        uid: effective_uid()?,
    });

    loop {
        let (stream, _) = listener.accept().await?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(state, stream).await {
                eprintln!("focal-vector local IPC error: {error}");
            }
        });
    }
}

struct LocalState {
    database: Arc<Database>,
    max_body_bytes: usize,
    max_batch_points: usize,
    max_k: usize,
    max_dimension: usize,
    max_ef_search: usize,
    admission: Arc<Semaphore>,
    uid: u32,
}

async fn handle_connection(state: Arc<LocalState>, mut stream: UnixStream) -> Result<()> {
    let credentials = stream
        .peer_cred()
        .map_err(|error| Error::Concurrency(format!("could not inspect IPC peer: {error}")))?;
    if credentials.uid() != state.uid {
        return Err(Error::Unauthorized);
    }

    let mut input = Vec::new();
    (&mut stream)
        .take(state.max_body_bytes as u64 + 1)
        .read_to_end(&mut input)
        .await?;
    let response = if input.len() > state.max_body_bytes {
        error_response(Error::ResourceExhausted(
            "local IPC request exceeds the configured body limit".into(),
        ))
    } else {
        dispatch_envelope(&state, &input).await
    };
    let output = serde_json::to_vec(&protocol::Envelope::current(response))
        .map_err(|error| Error::Concurrency(format!("could not encode IPC response: {error}")))?;
    stream.write_all(&output).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn dispatch_envelope(state: &Arc<LocalState>, input: &[u8]) -> protocol::Response {
    let envelope: protocol::Envelope<protocol::Request> = match serde_json::from_slice(input) {
        Ok(envelope) => envelope,
        Err(error) => {
            return error_response(Error::InvalidConfiguration(format!(
                "invalid local IPC request: {error}"
            )));
        }
    };
    if envelope.protocol_version != protocol::PROTOCOL_VERSION {
        return protocol::Response::Error {
            code: "unsupported_protocol".into(),
            message: format!(
                "client sent protocol version {}; server supports {}",
                envelope.protocol_version,
                protocol::PROTOCOL_VERSION
            ),
        };
    }
    match dispatch(state, envelope.payload).await {
        Ok(response) => response,
        Err(error) => error_response(error),
    }
}

async fn dispatch(
    state: &Arc<LocalState>,
    request: protocol::Request,
) -> Result<protocol::Response> {
    if matches!(request, protocol::Request::Hello) {
        return Ok(protocol::Response::Hello {
            server_version: env!("CARGO_PKG_VERSION").into(),
        });
    }
    let permit = Arc::clone(&state.admission)
        .try_acquire_owned()
        .map_err(|_| Error::ResourceExhausted("too many concurrent storage operations".into()))?;
    let state = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        dispatch_blocking(&state, request)
    })
    .await
    .map_err(|error| Error::Concurrency(format!("local IPC task failed: {error}")))?
}

fn dispatch_blocking(state: &LocalState, request: protocol::Request) -> Result<protocol::Response> {
    match request {
        protocol::Request::Hello => unreachable!("hello is handled without a blocking task"),
        protocol::Request::ListCollections => Ok(protocol::Response::Collections {
            collections: state
                .database
                .list_collections()?
                .into_iter()
                .map(|collection| protocol::CollectionInfo {
                    name: collection.name,
                    dimension: collection.config.dimension,
                    metric: metric_to_protocol(collection.config.metric),
                    points: collection.points,
                    latest_sequence: collection.latest_sequence,
                    pending_points: collection.pending_points,
                })
                .collect(),
        }),
        protocol::Request::CreateCollection {
            name,
            dimension,
            metric,
        } => {
            if dimension == 0 || dimension > state.max_dimension {
                return Err(Error::ResourceExhausted(format!(
                    "dimension must be between 1 and {}",
                    state.max_dimension
                )));
            }
            let metric = metric_from_protocol(metric);
            let collection = state
                .database
                .create_collection(&name, CollectionConfig { dimension, metric })?;
            Ok(protocol::Response::Created {
                collection: protocol::CollectionInfo {
                    name,
                    dimension,
                    metric: metric_to_protocol(metric),
                    points: 0,
                    latest_sequence: collection.latest_sequence()?,
                    pending_points: 0,
                },
            })
        }
        protocol::Request::Upsert { collection, points } => {
            if points.is_empty() || points.len() > state.max_batch_points {
                return Err(Error::ResourceExhausted(format!(
                    "upsert batch must contain 1-{} points",
                    state.max_batch_points
                )));
            }
            let points = points
                .into_iter()
                .map(point_from_protocol)
                .collect::<Result<Vec<_>>>()?;
            let sequence = state.database.collection(&collection)?.upsert(points)?;
            Ok(protocol::Response::Sequence { sequence })
        }
        protocol::Request::Delete { collection, ids } => {
            if ids.is_empty() || ids.len() > state.max_batch_points {
                return Err(Error::ResourceExhausted(format!(
                    "delete batch must contain 1-{} IDs",
                    state.max_batch_points
                )));
            }
            let sequence = state.database.collection(&collection)?.delete(ids)?;
            Ok(protocol::Response::Sequence { sequence })
        }
        protocol::Request::Query {
            collection,
            vector,
            k,
            filter,
            ef_search,
        } => {
            if k == 0 || k > state.max_k {
                return Err(Error::ResourceExhausted(format!(
                    "k must be between 1 and {}",
                    state.max_k
                )));
            }
            if ef_search.is_some_and(|value| value > state.max_ef_search) {
                return Err(Error::ResourceExhausted(format!(
                    "ef_search must not exceed {}",
                    state.max_ef_search
                )));
            }
            let filter = filter.map(filter_from_protocol).transpose()?;
            let collection = state.database.collection(&collection)?;
            let hits = match ef_search {
                Some(value) => collection.search_with_ef(vector, k, filter.as_ref(), value)?,
                None => collection.search(vector, k, filter.as_ref())?,
            };
            Ok(protocol::Response::Query {
                hits: hits
                    .into_iter()
                    .map(|hit| protocol::SearchHit {
                        id: hit.id,
                        score: hit.score,
                        metadata: hit
                            .metadata
                            .into_iter()
                            .map(|(key, value)| (key, json_value(value)))
                            .collect(),
                        sequence: hit.sequence,
                    })
                    .collect(),
            })
        }
        protocol::Request::Flush { collection } => {
            let sequence = state.database.collection(&collection)?.flush()?;
            Ok(protocol::Response::Sequence { sequence })
        }
    }
}

fn point_from_protocol(point: protocol::Point) -> Result<UpsertPoint> {
    Ok(UpsertPoint {
        id: point.id,
        vector: point.vector,
        metadata: point
            .metadata
            .into_iter()
            .map(|(key, value)| Ok((key, metadata_value(value)?)))
            .collect::<Result<BTreeMap<_, _>>>()?,
    })
}

fn filter_from_protocol(filter: protocol::Filter) -> Result<Filter> {
    let mut remaining = 1_024;
    filter_from_protocol_bounded(filter, 0, &mut remaining)
}

fn filter_from_protocol_bounded(
    filter: protocol::Filter,
    depth: usize,
    remaining: &mut usize,
) -> Result<Filter> {
    if depth >= 32 {
        return Err(Error::ResourceExhausted(
            "filter nesting exceeds 32 levels".into(),
        ));
    }
    *remaining = remaining
        .checked_sub(1)
        .ok_or_else(|| Error::ResourceExhausted("filter contains more than 1024 nodes".into()))?;
    Ok(match filter {
        protocol::Filter::MatchAll => Filter::MatchAll,
        protocol::Filter::Eq { field, value } => Filter::Eq {
            field: validated_field(field)?,
            value: metadata_value(value)?,
        },
        protocol::Filter::Range { field, gte, lt } => Filter::Range {
            field: validated_field(field)?,
            gte,
            lt,
        },
        protocol::Filter::And { filters } => Filter::And(
            filters
                .into_iter()
                .map(|filter| filter_from_protocol_bounded(filter, depth + 1, remaining))
                .collect::<Result<_>>()?,
        ),
        protocol::Filter::Or { filters } => Filter::Or(
            filters
                .into_iter()
                .map(|filter| filter_from_protocol_bounded(filter, depth + 1, remaining))
                .collect::<Result<_>>()?,
        ),
        protocol::Filter::Not { filter } => Filter::Not(Box::new(filter_from_protocol_bounded(
            *filter,
            depth + 1,
            remaining,
        )?)),
    })
}

fn metadata_value(value: JsonValue) -> Result<Value> {
    match value {
        JsonValue::String(value) => Ok(Value::Keyword(value)),
        JsonValue::Bool(value) => Ok(Value::Boolean(value)),
        JsonValue::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Value::Integer(value))
            } else {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(Value::Float)
                    .ok_or(Error::InvalidQuery("metadata number is out of range"))
            }
        }
        _ => Err(Error::InvalidQuery(
            "metadata values must be strings, numbers, or booleans",
        )),
    }
}

fn json_value(value: Value) -> JsonValue {
    match value {
        Value::Keyword(value) => JsonValue::String(value),
        Value::Integer(value) => JsonValue::from(value),
        Value::Float(value) => JsonValue::from(value),
        Value::Boolean(value) => JsonValue::from(value),
    }
}

fn validated_field(field: String) -> Result<String> {
    if field.is_empty() || field.len() > 128 {
        Err(Error::InvalidQuery(
            "filter field names must contain 1-128 bytes",
        ))
    } else {
        Ok(field)
    }
}

fn metric_from_protocol(metric: protocol::Metric) -> Metric {
    match metric {
        protocol::Metric::Cosine => Metric::Cosine,
        protocol::Metric::DotProduct => Metric::DotProduct,
        protocol::Metric::Euclidean => Metric::Euclidean,
    }
}

fn metric_to_protocol(metric: Metric) -> protocol::Metric {
    match metric {
        Metric::Cosine => protocol::Metric::Cosine,
        Metric::DotProduct => protocol::Metric::DotProduct,
        Metric::Euclidean => protocol::Metric::Euclidean,
    }
}

fn error_response(error: Error) -> protocol::Response {
    let code = match &error {
        Error::InvalidDimension { .. }
        | Error::InvalidVector(_)
        | Error::InvalidConfig(_)
        | Error::InvalidConfiguration(_)
        | Error::InvalidQuery(_) => "invalid_request",
        Error::NotFound(_) => "not_found",
        Error::AlreadyExists(_) => "already_exists",
        Error::Unauthorized => "unauthorized",
        Error::ResourceExhausted(_) => "resource_exhausted",
        Error::Io(_)
        | Error::CorruptStorage(_)
        | Error::Concurrency(_)
        | Error::SequenceOverflow => "internal",
    };
    protocol::Response::Error {
        code: code.into(),
        message: error.to_string(),
    }
}

fn validate_limits(config: &ServerConfig) -> Result<()> {
    if config.max_body_bytes == 0
        || config.max_batch_points == 0
        || config.max_k == 0
        || config.max_dimension == 0
        || config.max_ef_search == 0
        || config.max_concurrent_operations == 0
    {
        return Err(Error::InvalidConfig(
            "server request limits must be greater than zero",
        ));
    }
    Ok(())
}

fn prepare_socket_path(socket: &Path) -> Result<()> {
    let parent = socket.parent().ok_or(Error::InvalidConfig(
        "local IPC socket must have a parent directory",
    ))?;
    fs::create_dir_all(parent)?;
    let uid = effective_uid()?;
    let parent_metadata = fs::symlink_metadata(parent)?;
    if !parent_metadata.file_type().is_dir() || parent_metadata.uid() != uid {
        return Err(Error::Unauthorized);
    }
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;

    match fs::symlink_metadata(socket) {
        Ok(metadata) if metadata.file_type().is_socket() && metadata.uid() == uid => {
            if std::os::unix::net::UnixStream::connect(socket).is_ok() {
                return Err(Error::AlreadyExists(format!(
                    "active local IPC socket {}",
                    socket.display()
                )));
            }
            fs::remove_file(socket)?;
        }
        Ok(_) => {
            return Err(Error::InvalidConfiguration(format!(
                "refusing to replace non-socket or foreign-owned path {}",
                socket.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn effective_uid() -> Result<u32> {
    Ok(fs::metadata("/proc/self")?.uid())
}

struct SocketCleanup(PathBuf);

impl Drop for SocketCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
