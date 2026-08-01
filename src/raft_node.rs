use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use openraft::error::{
    InstallSnapshotError, NetworkError, RPCError, RaftError, RemoteError, Unreachable,
};
use openraft::network::{RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use openraft::{BasicNode, Config, Raft};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::{
    CollectionConfig, DurableRaftLog, DurableShardStateMachine, Error, Filter, FocalRaftConfig,
    NodeId, Result, SearchHit, ShardCommand, ShardResponse,
};

pub type FocalRaft = Raft<FocalRaftConfig>;
type StandardRpcError = RPCError<NodeId, BasicNode, RaftError<NodeId>>;
type SnapshotRpcError = RPCError<NodeId, BasicNode, RaftError<NodeId, InstallSnapshotError>>;
const MAX_CLIENT_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_IN_FLIGHT_CLIENT_OPERATIONS: usize = 256;

#[derive(Clone)]
pub struct HttpRaftNetwork {
    client: reqwest::Client,
    token: Arc<str>,
    admission: Arc<Semaphore>,
    blocked: Arc<RwLock<BTreeSet<NodeId>>>,
}

impl HttpRaftNetwork {
    pub fn new(token: impl Into<String>) -> Result<Self> {
        let token = token.into();
        if token.is_empty() {
            return Err(Error::InvalidConfig("Raft peer token must not be empty"));
        }
        let client = reqwest::Client::builder()
            .build()
            .map_err(|error| Error::Concurrency(error.to_string()))?;
        Ok(Self {
            client,
            token: Arc::from(token),
            admission: Arc::new(Semaphore::new(MAX_IN_FLIGHT_CLIENT_OPERATIONS)),
            blocked: Arc::new(RwLock::new(BTreeSet::new())),
        })
    }

    pub fn set_blocked(&self, target: NodeId, blocked: bool) -> Result<()> {
        let mut targets = self
            .blocked
            .write()
            .map_err(|_| Error::Concurrency("Raft network fault lock is poisoned".into()))?;
        if blocked {
            targets.insert(target);
        } else {
            targets.remove(&target);
        }
        Ok(())
    }

    async fn send<Req, Resp, Remote>(
        &self,
        target: NodeId,
        node: &BasicNode,
        path: &str,
        request: &Req,
    ) -> std::result::Result<Resp, RPCError<NodeId, BasicNode, Remote>>
    where
        Req: Serialize + ?Sized,
        Resp: DeserializeOwned,
        Remote: std::error::Error + DeserializeOwned,
    {
        if self
            .blocked
            .read()
            .map_err(|_| {
                let error = std::io::Error::other("Raft network fault lock is poisoned");
                RPCError::Network(NetworkError::new(&error))
            })?
            .contains(&target)
        {
            let error = std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                format!("peer {target} is blocked by fault injection"),
            );
            return Err(RPCError::Unreachable(Unreachable::new(&error)));
        }
        let base = if node.addr.starts_with("http://") || node.addr.starts_with("https://") {
            node.addr.trim_end_matches('/').to_owned()
        } else {
            format!("http://{}", node.addr.trim_end_matches('/'))
        };
        let url = format!("{base}{path}");
        let response = self
            .client
            .post(url)
            .header("x-focal-raft-token", self.token.as_ref())
            .json(request)
            .send()
            .await
            .map_err(|error| {
                if error.is_connect() || error.is_timeout() {
                    RPCError::Unreachable(Unreachable::new(&error))
                } else {
                    RPCError::Network(NetworkError::new(&error))
                }
            })?;
        if !response.status().is_success() {
            let error = std::io::Error::other(format!("peer returned HTTP {}", response.status()));
            return Err(RPCError::Network(NetworkError::new(&error)));
        }
        let result: std::result::Result<Resp, Remote> = response
            .json()
            .await
            .map_err(|error| RPCError::Network(NetworkError::new(&error)))?;
        result.map_err(|error| RPCError::RemoteError(RemoteError::new(target, error)))
    }
}

impl RaftNetworkFactory<FocalRaftConfig> for HttpRaftNetwork {
    type Network = HttpRaftConnection;

    async fn new_client(&mut self, target: NodeId, node: &BasicNode) -> Self::Network {
        HttpRaftConnection {
            network: self.clone(),
            target,
            node: node.clone(),
        }
    }
}

pub struct HttpRaftConnection {
    network: HttpRaftNetwork,
    target: NodeId,
    node: BasicNode,
}

impl RaftNetwork<FocalRaftConfig> for HttpRaftConnection {
    async fn append_entries(
        &mut self,
        request: AppendEntriesRequest<FocalRaftConfig>,
        _option: RPCOption,
    ) -> std::result::Result<AppendEntriesResponse<NodeId>, StandardRpcError> {
        self.network
            .send(self.target, &self.node, "/internal/raft/append", &request)
            .await
    }

    async fn install_snapshot(
        &mut self,
        request: InstallSnapshotRequest<FocalRaftConfig>,
        _option: RPCOption,
    ) -> std::result::Result<InstallSnapshotResponse<NodeId>, SnapshotRpcError> {
        self.network
            .send(self.target, &self.node, "/internal/raft/snapshot", &request)
            .await
    }

    async fn vote(
        &mut self,
        request: VoteRequest<NodeId>,
        _option: RPCOption,
    ) -> std::result::Result<VoteResponse<NodeId>, StandardRpcError> {
        self.network
            .send(self.target, &self.node, "/internal/raft/vote", &request)
            .await
    }
}

#[derive(Clone)]
pub struct RaftNode {
    id: NodeId,
    raft: FocalRaft,
    state_machine: DurableShardStateMachine,
    network: HttpRaftNetwork,
    token: Arc<str>,
}

impl RaftNode {
    pub async fn open(
        id: NodeId,
        directory: impl AsRef<Path>,
        collection_config: CollectionConfig,
        peer_token: impl Into<String>,
        raft_config: Config,
    ) -> Result<Self> {
        let token = peer_token.into();
        persist_node_id(directory.as_ref(), id)?;
        let network = HttpRaftNetwork::new(token.clone())?;
        let log_store = DurableRaftLog::open(directory.as_ref().join("log"))?;
        let state_machine =
            DurableShardStateMachine::open(directory.as_ref().join("state"), collection_config)?;
        let config = Arc::new(
            raft_config
                .validate()
                .map_err(|error| Error::InvalidConfiguration(error.to_string()))?,
        );
        let raft = Raft::new(
            id,
            config,
            network.clone(),
            log_store,
            state_machine.clone(),
        )
        .await
        .map_err(|error| Error::Concurrency(error.to_string()))?;
        Ok(Self {
            id,
            raft,
            state_machine,
            network,
            token: Arc::from(token),
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn raft(&self) -> &FocalRaft {
        &self.raft
    }

    pub fn set_peer_blocked(&self, target: NodeId, blocked: bool) -> Result<()> {
        self.network.set_blocked(target, blocked)
    }

    pub async fn initialize(&self, members: BTreeMap<NodeId, BasicNode>) -> Result<()> {
        self.raft
            .initialize(members)
            .await
            .map_err(|error| Error::Concurrency(error.to_string()))
    }

    pub async fn write(&self, command: ShardCommand) -> Result<ShardResponse> {
        let _permit =
            self.network.admission.try_acquire().map_err(|_| {
                Error::ResourceExhausted("Raft client operation limit reached".into())
            })?;
        self.raft
            .client_write(command)
            .await
            .map(|response| response.data)
            .map_err(|error| Error::Concurrency(error.to_string()))
    }

    pub async fn add_learner(&self, id: NodeId, node: BasicNode) -> Result<()> {
        self.raft
            .add_learner(id, node, true)
            .await
            .map(|_| ())
            .map_err(|error| Error::Concurrency(error.to_string()))
    }

    pub async fn change_membership(&self, voters: BTreeSet<NodeId>, retain: bool) -> Result<()> {
        self.raft
            .change_membership(voters, retain)
            .await
            .map(|_| ())
            .map_err(|error| Error::Concurrency(error.to_string()))
    }

    pub async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchHit>> {
        let _permit =
            self.network.admission.try_acquire().map_err(|_| {
                Error::ResourceExhausted("Raft client operation limit reached".into())
            })?;
        self.raft
            .ensure_linearizable()
            .await
            .map_err(|error| Error::Concurrency(error.to_string()))?;
        self.state_machine.search(query, k, filter).await
    }

    pub async fn local_len(&self) -> usize {
        self.state_machine.len().await
    }
}

fn persist_node_id(directory: &Path, id: NodeId) -> Result<()> {
    fs::create_dir_all(directory)?;
    let path = directory.join("node.id");
    if path.exists() {
        let stored = fs::read_to_string(path)?;
        let stored = stored
            .trim()
            .parse::<NodeId>()
            .map_err(|error| Error::CorruptStorage(error.to_string()))?;
        if stored != id {
            return Err(Error::CorruptStorage(format!(
                "Raft node ID changed from {stored} to {id}"
            )));
        }
        return Ok(());
    }
    let temporary = directory.join(format!(".node.id.tmp-{}", std::process::id()));
    fs::write(&temporary, format!("{id}\n"))?;
    fs::File::open(&temporary)?.sync_all()?;
    fs::rename(temporary, path)?;
    fs::File::open(directory)?.sync_all()?;
    Ok(())
}

pub fn raft_router(node: Arc<RaftNode>) -> Router {
    let internal = Router::new()
        .route("/internal/raft/vote", post(vote))
        .route("/internal/raft/append", post(append))
        .route("/internal/raft/snapshot", post(snapshot))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024));
    let client = Router::new()
        .route("/v1/raft/initialize", post(initialize_cluster))
        .route("/v1/raft/write", post(client_write))
        .route("/v1/raft/query", post(linearizable_query))
        .route("/v1/raft/learners/{id}", post(add_learner))
        .route("/v1/raft/membership", post(change_membership))
        .route("/v1/raft/status", get(raft_status))
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .layer(DefaultBodyLimit::max(MAX_CLIENT_BODY_BYTES));
    internal.merge(client).with_state(node)
}

#[derive(Debug, Deserialize)]
struct InitializeRequest {
    members: BTreeMap<NodeId, String>,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    vector: Vec<f32>,
    k: usize,
    filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
struct LearnerRequest {
    address: String,
}

#[derive(Debug, Deserialize)]
struct MembershipRequest {
    voters: BTreeSet<NodeId>,
    #[serde(default)]
    retain_removed_as_learners: bool,
}

type ApiFailure = (StatusCode, Json<serde_json::Value>);

fn api_error(error: Error) -> ApiFailure {
    let status = match error {
        Error::InvalidConfig(_)
        | Error::InvalidConfiguration(_)
        | Error::InvalidDimension { .. }
        | Error::InvalidQuery(_)
        | Error::InvalidVector(_) => StatusCode::BAD_REQUEST,
        Error::ResourceExhausted(_) => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    };
    (
        status,
        Json(serde_json::json!({ "error": error.to_string() })),
    )
}

fn unauthorized() -> ApiFailure {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({ "error": "unauthorized" })),
    )
}

async fn initialize_cluster(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<InitializeRequest>,
) -> std::result::Result<StatusCode, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    let members = request
        .members
        .into_iter()
        .map(|(id, address)| (id, BasicNode::new(address)))
        .collect();
    node.initialize(members).await.map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn client_write(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(command): Json<ShardCommand>,
) -> std::result::Result<Json<ShardResponse>, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    node.write(command).await.map(Json).map_err(api_error)
}

async fn linearizable_query(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<QueryRequest>,
) -> std::result::Result<Json<Vec<SearchHit>>, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    node.search(request.vector, request.k, request.filter.as_ref())
        .await
        .map(Json)
        .map_err(api_error)
}

async fn add_learner(
    State(node): State<Arc<RaftNode>>,
    AxumPath(id): AxumPath<NodeId>,
    headers: HeaderMap,
    Json(request): Json<LearnerRequest>,
) -> std::result::Result<StatusCode, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    node.add_learner(id, BasicNode::new(request.address))
        .await
        .map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn change_membership(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<MembershipRequest>,
) -> std::result::Result<StatusCode, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    node.change_membership(request.voters, request.retain_removed_as_learners)
        .await
        .map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn raft_status(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
) -> std::result::Result<Json<openraft::RaftMetrics<NodeId, BasicNode>>, ApiFailure> {
    if !authorized(&headers, &node.token) {
        return Err(unauthorized());
    }
    Ok(Json(node.raft.metrics().borrow().clone()))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(node): State<Arc<RaftNode>>) -> StatusCode {
    if node.raft.metrics().borrow().running_state.is_ok() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

fn authorized(headers: &HeaderMap, token: &str) -> bool {
    headers
        .get("x-focal-raft-token")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|provided| constant_time_equal(provided.as_bytes(), token.as_bytes()))
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

async fn vote(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<VoteRequest<NodeId>>,
) -> std::result::Result<
    Json<std::result::Result<VoteResponse<NodeId>, RaftError<NodeId>>>,
    StatusCode,
> {
    if !authorized(&headers, &node.token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(Json(node.raft.vote(request).await))
}

async fn append(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<AppendEntriesRequest<FocalRaftConfig>>,
) -> std::result::Result<
    Json<std::result::Result<AppendEntriesResponse<NodeId>, RaftError<NodeId>>>,
    StatusCode,
> {
    if !authorized(&headers, &node.token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(Json(node.raft.append_entries(request).await))
}

async fn snapshot(
    State(node): State<Arc<RaftNode>>,
    headers: HeaderMap,
    Json(request): Json<InstallSnapshotRequest<FocalRaftConfig>>,
) -> std::result::Result<
    Json<
        std::result::Result<
            InstallSnapshotResponse<NodeId>,
            RaftError<NodeId, InstallSnapshotError>,
        >,
    >,
    StatusCode,
> {
    if !authorized(&headers, &node.token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(Json(node.raft.install_snapshot(request).await))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use crate::{Metric, UpsertPoint};

    use super::*;

    fn directory() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("focal-vector-node-{}-{nonce}", std::process::id()))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_node_commits_linearizable_write_and_recovers() {
        let directory = directory();
        let config = CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        };
        let raft_config = Config {
            cluster_name: "single-node-test".into(),
            heartbeat_interval: 50,
            election_timeout_min: 150,
            election_timeout_max: 300,
            ..Default::default()
        };
        let node = RaftNode::open(1, &directory, config, "secret", raft_config)
            .await
            .unwrap();
        node.initialize(BTreeMap::from([(1, BasicNode::new("127.0.0.1:1"))]))
            .await
            .unwrap();
        node.raft
            .wait(Some(Duration::from_secs(3)))
            .state(openraft::ServerState::Leader, "leader")
            .await
            .unwrap();
        let response = node
            .write(ShardCommand::Upsert {
                client_id: "client".into(),
                request_id: 1,
                points: vec![UpsertPoint {
                    id: "p".into(),
                    vector: vec![1.0, 0.0],
                    metadata: BTreeMap::new(),
                }],
            })
            .await
            .unwrap();
        assert_eq!(response, ShardResponse::Applied { sequence: 1 });
        assert_eq!(
            node.search(vec![1.0, 0.0], 1, None).await.unwrap()[0].id,
            "p"
        );
        node.raft.shutdown().await.unwrap();
        drop(node);

        let reopened = DurableShardStateMachine::open(directory.join("state"), config).unwrap();
        assert_eq!(reopened.len().await, 1);
        drop(reopened);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 6)]
    async fn three_node_cluster_replicates_to_a_majority() {
        let root = directory();
        let config = CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        };
        let mut listeners = Vec::new();
        let mut members = BTreeMap::new();
        for id in 1..=3 {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap().to_string();
            members.insert(id, BasicNode::new(address));
            listeners.push(listener);
        }

        let mut nodes = Vec::new();
        for id in 1..=3 {
            let raft_config = Config {
                cluster_name: "three-node-test".into(),
                heartbeat_interval: 50,
                election_timeout_min: 200,
                election_timeout_max: 400,
                ..Default::default()
            };
            nodes.push(Arc::new(
                RaftNode::open(
                    id,
                    root.join(format!("node-{id}")),
                    config,
                    "cluster-secret",
                    raft_config,
                )
                .await
                .unwrap(),
            ));
        }

        let mut servers = Vec::new();
        for (listener, node) in listeners.into_iter().zip(nodes.iter().cloned()) {
            servers.push(tokio::spawn(async move {
                axum::serve(listener, raft_router(node)).await.unwrap();
            }));
        }
        nodes[0].initialize(members.clone()).await.unwrap();
        nodes[0]
            .raft
            .wait(Some(Duration::from_secs(5)))
            .state(openraft::ServerState::Leader, "initial leader")
            .await
            .unwrap();
        let response = nodes[0]
            .write(ShardCommand::Upsert {
                client_id: "client".into(),
                request_id: 9,
                points: vec![UpsertPoint {
                    id: "replicated".into(),
                    vector: vec![1.0, 0.0],
                    metadata: BTreeMap::new(),
                }],
            })
            .await
            .unwrap();
        assert_eq!(response, ShardResponse::Applied { sequence: 1 });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let applied = nodes[1].local_len().await + nodes[2].local_len().await;
            if applied >= 1 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "followers did not apply"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        nodes[0].set_peer_blocked(2, true).unwrap();
        nodes[0].set_peer_blocked(3, true).unwrap();
        nodes[1].set_peer_blocked(1, true).unwrap();
        nodes[2].set_peer_blocked(1, true).unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        let response = loop {
            let command = || ShardCommand::Upsert {
                client_id: "client".into(),
                request_id: 10,
                points: vec![UpsertPoint {
                    id: "during-partition".into(),
                    vector: vec![2.0, 0.0],
                    metadata: BTreeMap::new(),
                }],
            };
            if let Ok(response) = nodes[1].write(command()).await {
                break response;
            }
            if let Ok(response) = nodes[2].write(command()).await {
                break response;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "majority partition did not elect a leader"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(response, ShardResponse::Applied { sequence: 2 });

        let isolated_write = tokio::time::timeout(
            Duration::from_secs(1),
            nodes[0].write(ShardCommand::Upsert {
                client_id: "isolated-client".into(),
                request_id: 1,
                points: vec![UpsertPoint {
                    id: "must-not-ack".into(),
                    vector: vec![9.0, 9.0],
                    metadata: BTreeMap::new(),
                }],
            }),
        )
        .await;
        assert!(
            !matches!(isolated_write, Ok(Ok(_))),
            "isolated leader acknowledged a write without quorum"
        );
        nodes[1].set_peer_blocked(1, false).unwrap();
        nodes[2].set_peer_blocked(1, false).unwrap();

        nodes[0].raft.shutdown().await.unwrap();
        servers[0].abort();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        let response = loop {
            let command = || ShardCommand::Upsert {
                client_id: "client".into(),
                request_id: 11,
                points: vec![UpsertPoint {
                    id: "after-failover".into(),
                    vector: vec![0.0, 1.0],
                    metadata: BTreeMap::new(),
                }],
            };
            if let Ok(response) = nodes[1].write(command()).await {
                break response;
            }
            if let Ok(response) = nodes[2].write(command()).await {
                break response;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "surviving nodes did not elect a leader"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(response, ShardResponse::Applied { sequence: 3 });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while nodes[1].local_len().await < 3 || nodes[2].local_len().await < 3 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "post-failover write did not reach both survivors"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        for node in &nodes[1..] {
            node.raft.shutdown().await.unwrap();
        }
        for server in servers {
            server.abort();
            let _ = server.await;
        }
        drop(nodes);

        let mut restarted = Vec::new();
        let mut restarted_servers = Vec::new();
        for id in 1..=3 {
            let address = &members[&id].addr;
            let listener = tokio::net::TcpListener::bind(address).await.unwrap();
            let raft_config = Config {
                cluster_name: "three-node-test".into(),
                heartbeat_interval: 50,
                election_timeout_min: 200,
                election_timeout_max: 400,
                ..Default::default()
            };
            let node = Arc::new(
                RaftNode::open(
                    id,
                    root.join(format!("node-{id}")),
                    config,
                    "cluster-secret",
                    raft_config,
                )
                .await
                .unwrap(),
            );
            restarted_servers.push(tokio::spawn({
                let node = Arc::clone(&node);
                async move {
                    axum::serve(listener, raft_router(node)).await.unwrap();
                }
            }));
            restarted.push(node);
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        let response = loop {
            let mut committed = None;
            for node in &restarted {
                let command = ShardCommand::Upsert {
                    client_id: "client".into(),
                    request_id: 12,
                    points: vec![UpsertPoint {
                        id: "after-restart".into(),
                        vector: vec![1.0, 1.0],
                        metadata: BTreeMap::new(),
                    }],
                };
                if let Ok(response) = node.write(command).await {
                    committed = Some(response);
                    break;
                }
            }
            if let Some(response) = committed {
                break response;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "restarted voters did not elect a leader"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(response, ShardResponse::Applied { sequence: 4 });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let mut all_applied = true;
            for node in &restarted {
                all_applied &= node.local_len().await == 4;
            }
            if all_applied {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "restarted voters did not converge"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        loop {
            let mut changed = false;
            for node in &restarted {
                if node
                    .change_membership(BTreeSet::from([1, 2]), false)
                    .await
                    .is_ok()
                {
                    changed = true;
                    break;
                }
            }
            if changed {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "membership change did not commit"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        let response = loop {
            let mut committed = None;
            for node in &restarted[..2] {
                let command = ShardCommand::Upsert {
                    client_id: "client".into(),
                    request_id: 13,
                    points: vec![UpsertPoint {
                        id: "after-membership-change".into(),
                        vector: vec![2.0, 2.0],
                        metadata: BTreeMap::new(),
                    }],
                };
                if let Ok(response) = node.write(command).await {
                    committed = Some(response);
                    break;
                }
            }
            if let Some(response) = committed {
                break response;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "new voter set did not commit a write"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(response, ShardResponse::Applied { sequence: 5 });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while restarted[0].local_len().await < 5 || restarted[1].local_len().await < 5 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "new voter set did not converge"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        for node in &restarted {
            node.raft.shutdown().await.unwrap();
        }
        for server in restarted_servers {
            server.abort();
        }
        drop(restarted);
        fs::remove_dir_all(root).unwrap();
    }
}
