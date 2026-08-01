use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::Semaphore;

use crate::{
    CollectionConfig, Database, Error, Filter, Metric, Result, SearchHit, UpsertPoint, Value,
};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bearer_token: Option<String>,
    pub max_body_bytes: usize,
    pub max_batch_points: usize,
    pub max_k: usize,
    pub max_dimension: usize,
    pub max_ef_search: usize,
    pub max_concurrent_operations: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bearer_token: None,
            max_body_bytes: 16 * 1024 * 1024,
            max_batch_points: 1_000,
            max_k: 1_000,
            max_dimension: 4_096,
            max_ef_search: 4_096,
            max_concurrent_operations: 64,
        }
    }
}

#[derive(Clone)]
struct AppState {
    database: Arc<Database>,
    token: Option<Arc<str>>,
    max_batch_points: usize,
    max_k: usize,
    max_dimension: usize,
    max_ef_search: usize,
    admission: Arc<Semaphore>,
    metrics: Arc<ServiceMetrics>,
}

#[derive(Default)]
struct ServiceMetrics {
    requests: AtomicU64,
    errors: AtomicU64,
    queries: AtomicU64,
    query_microseconds: AtomicU64,
    upserted_points: AtomicU64,
    deleted_ids: AtomicU64,
    flushes: AtomicU64,
    backups: AtomicU64,
    restores: AtomicU64,
}

pub fn router(database: Arc<Database>, config: ServerConfig) -> Result<Router> {
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
    let state = AppState {
        database,
        token: config.bearer_token.map(Arc::from),
        max_batch_points: config.max_batch_points,
        max_k: config.max_k,
        max_dimension: config.max_dimension,
        max_ef_search: config.max_ef_search,
        admission: Arc::new(Semaphore::new(config.max_concurrent_operations)),
        metrics: Arc::new(ServiceMetrics::default()),
    };
    let observer_state = state.clone();
    Ok(Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(readiness))
        .route("/metrics", get(metrics))
        .route("/v1/collections", get(list_collections))
        .route("/v1/backups", get(list_backups))
        .route("/v1/collections/{name}", put(create_collection))
        .route("/v1/collections/{name}/points/upsert", post(upsert_points))
        .route("/v1/collections/{name}/points/delete", post(delete_points))
        .route("/v1/collections/{name}/query", post(query))
        .route("/v1/collections/{name}/flush", post(flush))
        .route(
            "/v1/collections/{name}/backups/{backup}",
            post(backup_collection),
        )
        .route(
            "/v1/backups/{backup}/restore/{name}",
            post(restore_collection),
        )
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(middleware::from_fn_with_state(observer_state, observe))
        .with_state(state))
}

async fn observe(State(state): State<AppState>, request: Request, next: Next) -> Response {
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let response = next.run(request).await;
    if response.status().is_client_error() || response.status().is_server_error() {
        state.metrics.errors.fetch_add(1, Ordering::Relaxed);
    }
    response
}

async fn health() -> Json<StatusResponse> {
    Json(StatusResponse { status: "ok" })
}

async fn readiness(State(state): State<AppState>) -> impl IntoResponse {
    if state.database.is_ready() {
        (StatusCode::OK, Json(StatusResponse { status: "ready" }))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(StatusResponse {
                status: "not_ready",
            }),
        )
    }
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = &state.metrics;
    let body = format!(
        concat!(
            "# TYPE focal_requests_total counter\n",
            "focal_requests_total {}\n",
            "# TYPE focal_errors_total counter\n",
            "focal_errors_total {}\n",
            "# TYPE focal_queries_total counter\n",
            "focal_queries_total {}\n",
            "# TYPE focal_query_seconds_total counter\n",
            "focal_query_seconds_total {:.6}\n",
            "# TYPE focal_upserted_points_total counter\n",
            "focal_upserted_points_total {}\n",
            "# TYPE focal_deleted_ids_total counter\n",
            "focal_deleted_ids_total {}\n",
            "# TYPE focal_flushes_total counter\n",
            "focal_flushes_total {}\n",
            "# TYPE focal_backups_total counter\n",
            "focal_backups_total {}\n",
            "# TYPE focal_restores_total counter\n",
            "focal_restores_total {}\n"
        ),
        metrics.requests.load(Ordering::Relaxed),
        metrics.errors.load(Ordering::Relaxed),
        metrics.queries.load(Ordering::Relaxed),
        metrics.query_microseconds.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        metrics.upserted_points.load(Ordering::Relaxed),
        metrics.deleted_ids.load(Ordering::Relaxed),
        metrics.flushes.load(Ordering::Relaxed),
        metrics.backups.load(Ordering::Relaxed),
        metrics.restores.load(Ordering::Relaxed),
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}

async fn list_backups(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<ListBackupsResponse>> {
    authorize(&state, &headers)?;
    let database = Arc::clone(&state.database);
    let backups = blocking(&state, move || database.list_backups()).await?;
    Ok(Json(ListBackupsResponse { backups }))
}

async fn list_collections(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> ApiResult<Json<ListCollectionsResponse>> {
    authorize(&state, &headers)?;
    let database = Arc::clone(&state.database);
    let collections = blocking(&state, move || database.list_collections()).await?;
    Ok(Json(ListCollectionsResponse {
        collections: collections
            .into_iter()
            .map(|summary| CollectionResponse {
                name: summary.name,
                dimension: summary.config.dimension,
                metric: ApiMetric::from(summary.config.metric),
                points: summary.points,
                latest_sequence: summary.latest_sequence,
                pending_points: summary.pending_points,
            })
            .collect(),
    }))
}

async fn create_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(request): Json<CreateCollectionRequest>,
) -> ApiResult<(StatusCode, Json<SequenceResponse>)> {
    authorize(&state, &headers)?;
    if request.dimension == 0 || request.dimension > state.max_dimension {
        return Err(ApiError::from(Error::ResourceExhausted(format!(
            "dimension must be between 1 and {}",
            state.max_dimension
        ))));
    }
    let database = Arc::clone(&state.database);
    blocking(&state, move || {
        database.create_collection(
            &name,
            CollectionConfig {
                dimension: request.dimension,
                metric: request.metric.into(),
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(SequenceResponse { sequence: 0 })))
}

async fn upsert_points(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(request): Json<UpsertRequest>,
) -> ApiResult<Json<SequenceResponse>> {
    authorize(&state, &headers)?;
    if request.points.is_empty() || request.points.len() > state.max_batch_points {
        return Err(ApiError::from(Error::ResourceExhausted(format!(
            "upsert batch must contain 1-{} points",
            state.max_batch_points
        ))));
    }
    let count = request.points.len() as u64;
    let points = request
        .points
        .into_iter()
        .map(UpsertPoint::try_from)
        .collect::<Result<Vec<_>>>()?;
    let database = Arc::clone(&state.database);
    let sequence = blocking(&state, move || database.collection(&name)?.upsert(points)).await?;
    state
        .metrics
        .upserted_points
        .fetch_add(count, Ordering::Relaxed);
    Ok(Json(SequenceResponse { sequence }))
}

async fn delete_points(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(request): Json<DeleteRequest>,
) -> ApiResult<Json<SequenceResponse>> {
    authorize(&state, &headers)?;
    if request.ids.is_empty() || request.ids.len() > state.max_batch_points {
        return Err(ApiError::from(Error::ResourceExhausted(format!(
            "delete batch must contain 1-{} IDs",
            state.max_batch_points
        ))));
    }
    let count = request.ids.len() as u64;
    let database = Arc::clone(&state.database);
    let sequence = blocking(&state, move || {
        database.collection(&name)?.delete(request.ids)
    })
    .await?;
    state
        .metrics
        .deleted_ids
        .fetch_add(count, Ordering::Relaxed);
    Ok(Json(SequenceResponse { sequence }))
}

async fn query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<QueryResponse>> {
    authorize(&state, &headers)?;
    if request.k == 0 || request.k > state.max_k {
        return Err(ApiError::from(Error::ResourceExhausted(format!(
            "k must be between 1 and {}",
            state.max_k
        ))));
    }
    if request
        .ef_search
        .is_some_and(|ef_search| ef_search > state.max_ef_search)
    {
        return Err(ApiError::from(Error::ResourceExhausted(format!(
            "ef_search must not exceed {}",
            state.max_ef_search
        ))));
    }
    let filter = request
        .filter
        .map(|filter| filter.into_filter(0))
        .transpose()?;
    let database = Arc::clone(&state.database);
    let started = Instant::now();
    let hits = blocking(&state, move || {
        let collection = database.collection(&name)?;
        match request.ef_search {
            Some(ef) => collection.search_with_ef(request.vector, request.k, filter.as_ref(), ef),
            None => collection.search(request.vector, request.k, filter.as_ref()),
        }
    })
    .await?;
    state.metrics.queries.fetch_add(1, Ordering::Relaxed);
    state.metrics.query_microseconds.fetch_add(
        started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    Ok(Json(QueryResponse {
        hits: hits.into_iter().map(HitResponse::from).collect(),
    }))
}

async fn flush(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> ApiResult<Json<SequenceResponse>> {
    authorize(&state, &headers)?;
    let database = Arc::clone(&state.database);
    let sequence = blocking(&state, move || database.collection(&name)?.flush()).await?;
    state.metrics.flushes.fetch_add(1, Ordering::Relaxed);
    Ok(Json(SequenceResponse { sequence }))
}

async fn backup_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((name, backup)): Path<(String, String)>,
) -> ApiResult<(StatusCode, Json<SequenceResponse>)> {
    authorize(&state, &headers)?;
    let database = Arc::clone(&state.database);
    let sequence = blocking(&state, move || database.backup_collection(&name, &backup)).await?;
    state.metrics.backups.fetch_add(1, Ordering::Relaxed);
    Ok((StatusCode::CREATED, Json(SequenceResponse { sequence })))
}

async fn restore_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((backup, name)): Path<(String, String)>,
) -> ApiResult<(StatusCode, Json<SequenceResponse>)> {
    authorize(&state, &headers)?;
    let database = Arc::clone(&state.database);
    let collection = blocking(&state, move || database.restore_collection(&backup, &name)).await?;
    let sequence = collection.latest_sequence()?;
    state.metrics.restores.fetch_add(1, Ordering::Relaxed);
    Ok((StatusCode::CREATED, Json(SequenceResponse { sequence })))
}

async fn blocking<T: Send + 'static>(
    state: &AppState,
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let permit = Arc::clone(&state.admission)
        .try_acquire_owned()
        .map_err(|_| Error::ResourceExhausted("too many concurrent storage operations".into()))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation()
    })
    .await
    .map_err(|error| Error::Concurrency(format!("blocking task failed: {error}")))?
}

fn authorize(state: &AppState, headers: &HeaderMap) -> ApiResult<()> {
    let Some(expected) = &state.token else {
        return Ok(());
    };
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if supplied.is_some_and(|value| constant_time_equal(value.as_bytes(), expected.as_bytes())) {
        Ok(())
    } else {
        Err(ApiError::from(Error::Unauthorized))
    }
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

type ApiResult<T> = std::result::Result<T, ApiError>;

struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            Error::InvalidDimension { .. }
            | Error::InvalidVector(_)
            | Error::InvalidConfig(_)
            | Error::InvalidConfiguration(_)
            | Error::InvalidQuery(_) => StatusCode::BAD_REQUEST,
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::AlreadyExists(_) => StatusCode::CONFLICT,
            Error::Unauthorized => StatusCode::UNAUTHORIZED,
            Error::ResourceExhausted(_) => StatusCode::TOO_MANY_REQUESTS,
            Error::Io(_)
            | Error::CorruptStorage(_)
            | Error::Concurrency(_)
            | Error::SequenceOverflow => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
}

#[derive(Debug, Deserialize)]
struct CreateCollectionRequest {
    dimension: usize,
    metric: ApiMetric,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ApiMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl From<ApiMetric> for Metric {
    fn from(metric: ApiMetric) -> Self {
        match metric {
            ApiMetric::Cosine => Self::Cosine,
            ApiMetric::DotProduct => Self::DotProduct,
            ApiMetric::Euclidean => Self::Euclidean,
        }
    }
}

impl From<Metric> for ApiMetric {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Cosine => Self::Cosine,
            Metric::DotProduct => Self::DotProduct,
            Metric::Euclidean => Self::Euclidean,
        }
    }
}

#[derive(Debug, Deserialize)]
struct UpsertRequest {
    points: Vec<ApiPoint>,
}

#[derive(Debug, Deserialize)]
struct ApiPoint {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: BTreeMap<String, JsonValue>,
}

impl TryFrom<ApiPoint> for UpsertPoint {
    type Error = Error;

    fn try_from(point: ApiPoint) -> Result<Self> {
        let metadata = point
            .metadata
            .into_iter()
            .map(|(key, value)| Ok((key, metadata_value(value)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            id: point.id,
            vector: point.vector,
            metadata,
        })
    }
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

#[derive(Debug, Deserialize)]
struct DeleteRequest {
    ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    vector: Vec<f32>,
    k: usize,
    filter: Option<ApiFilter>,
    ef_search: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum ApiFilter {
    MatchAll,
    Eq {
        field: String,
        value: JsonValue,
    },
    Range {
        field: String,
        gte: Option<f64>,
        lt: Option<f64>,
    },
    And {
        filters: Vec<ApiFilter>,
    },
    Or {
        filters: Vec<ApiFilter>,
    },
    Not {
        filter: Box<ApiFilter>,
    },
}

impl ApiFilter {
    fn into_filter(self, depth: usize) -> Result<Filter> {
        let mut remaining = 1_024;
        self.into_filter_bounded(depth, &mut remaining)
    }

    fn into_filter_bounded(self, depth: usize, remaining: &mut usize) -> Result<Filter> {
        if depth >= 32 {
            return Err(Error::ResourceExhausted(
                "filter nesting exceeds 32 levels".into(),
            ));
        }
        *remaining = remaining.checked_sub(1).ok_or_else(|| {
            Error::ResourceExhausted("filter contains more than 1024 nodes".into())
        })?;
        Ok(match self {
            Self::MatchAll => Filter::MatchAll,
            Self::Eq { field, value } => Filter::Eq {
                field: validated_field(field)?,
                value: metadata_value(value)?,
            },
            Self::Range { field, gte, lt } => Filter::Range {
                field: validated_field(field)?,
                gte,
                lt,
            },
            Self::And { filters } => Filter::And(
                filters
                    .into_iter()
                    .map(|filter| filter.into_filter_bounded(depth + 1, remaining))
                    .collect::<Result<_>>()?,
            ),
            Self::Or { filters } => Filter::Or(
                filters
                    .into_iter()
                    .map(|filter| filter.into_filter_bounded(depth + 1, remaining))
                    .collect::<Result<_>>()?,
            ),
            Self::Not { filter } => {
                Filter::Not(Box::new(filter.into_filter_bounded(depth + 1, remaining)?))
            }
        })
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

#[derive(Serialize)]
struct SequenceResponse {
    sequence: u64,
}

#[derive(Serialize)]
struct ListCollectionsResponse {
    collections: Vec<CollectionResponse>,
}

#[derive(Serialize)]
struct ListBackupsResponse {
    backups: Vec<String>,
}

#[derive(Serialize)]
struct CollectionResponse {
    name: String,
    dimension: usize,
    metric: ApiMetric,
    points: usize,
    latest_sequence: u64,
    pending_points: usize,
}

#[derive(Serialize)]
struct QueryResponse {
    hits: Vec<HitResponse>,
}

#[derive(Serialize)]
struct HitResponse {
    id: String,
    score: f32,
    metadata: BTreeMap<String, JsonValue>,
    sequence: u64,
}

impl From<SearchHit> for HitResponse {
    fn from(hit: SearchHit) -> Self {
        Self {
            id: hit.id,
            score: hit.score,
            metadata: hit
                .metadata
                .into_iter()
                .map(|(key, value)| (key, json_value(value)))
                .collect(),
            sequence: hit.sequence,
        }
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

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn bearer_comparison_checks_contents_and_length() {
        assert!(constant_time_equal(b"secret", b"secret"));
        assert!(!constant_time_equal(b"secret", b"secrex"));
        assert!(!constant_time_equal(b"secret", b"secret-long"));
    }

    #[test]
    fn rejects_deep_filters_and_complex_metadata() {
        let mut filter = ApiFilter::MatchAll;
        for _ in 0..33 {
            filter = ApiFilter::Not {
                filter: Box::new(filter),
            };
        }
        assert!(matches!(
            filter.into_filter(0),
            Err(Error::ResourceExhausted(_))
        ));
        assert!(metadata_value(serde_json::json!({"nested": true})).is_err());
    }

    #[tokio::test]
    async fn authenticated_http_workflow_creates_upserts_and_queries() {
        let root = test_directory();
        let database = Arc::new(
            Database::open(
                &root,
                crate::DatabaseConfig {
                    flush_interval: Duration::from_secs(60),
                    dirty_point_threshold: 1_000,
                    ..crate::DatabaseConfig::default()
                },
            )
            .unwrap(),
        );
        let application = router(
            Arc::clone(&database),
            ServerConfig {
                bearer_token: Some("secret".into()),
                ..ServerConfig::default()
            },
        )
        .unwrap();

        let unauthorized = application
            .clone()
            .oneshot(json_request(
                "PUT",
                "/v1/collections/articles",
                r#"{"dimension":2,"metric":"cosine"}"#,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let created = application
            .clone()
            .oneshot(json_request(
                "PUT",
                "/v1/collections/articles",
                r#"{"dimension":2,"metric":"cosine"}"#,
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);

        let upserted = application
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/collections/articles/points/upsert",
                r#"{"points":[{"id":"doc-1","vector":[1.0,0.0],"metadata":{"tenant":"acme"}}]}"#,
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(upserted.status(), StatusCode::OK);

        let queried = application
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/collections/articles/query",
                r#"{"vector":[1.0,0.0],"k":1,"filter":{"op":"eq","field":"tenant","value":"acme"}}"#,
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(queried.status(), StatusCode::OK);
        let body = to_bytes(queried.into_body(), 64 * 1024).await.unwrap();
        let response: JsonValue = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["hits"][0]["id"], "doc-1");

        let backup = application
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/collections/articles/backups/snapshot-1",
                "",
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(backup.status(), StatusCode::CREATED);
        let restore = application
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/backups/snapshot-1/restore/articles-copy",
                "",
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(restore.status(), StatusCode::CREATED);
        let restored_query = application
            .clone()
            .oneshot(json_request(
                "POST",
                "/v1/collections/articles-copy/query",
                r#"{"vector":[1.0,0.0],"k":1}"#,
                Some("secret"),
            ))
            .await
            .unwrap();
        assert_eq!(restored_query.status(), StatusCode::OK);

        let metrics_response = application
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let metrics = to_bytes(metrics_response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let metrics = String::from_utf8(metrics.to_vec()).unwrap();
        assert!(metrics.contains("focal_queries_total 2"));
        assert!(metrics.contains("focal_backups_total 1"));
        assert!(metrics.contains("focal_restores_total 1"));

        drop(application);
        drop(database);
        fs::remove_dir_all(root).unwrap();
    }

    fn json_request(
        method: &str,
        uri: &str,
        body: &'static str,
        token: Option<&str>,
    ) -> Request<Body> {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(token) = token {
            request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        request.body(Body::from(body)).unwrap()
    }

    fn test_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("focal-vector-http-{}-{nonce}", std::process::id()))
    }
}
