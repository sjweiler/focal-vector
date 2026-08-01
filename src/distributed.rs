use std::collections::BTreeMap;

use serde::Serialize;
use tokio::task::JoinSet;

use crate::sharding::shard_index;
use crate::{Error, Filter, Result, SearchHit, ShardCommand, ShardResponse, UpsertPoint};

#[derive(Debug, Clone)]
pub struct ReplicaSet {
    pub addresses: Vec<String>,
}

#[derive(Clone)]
pub struct DistributedCollection {
    client: reqwest::Client,
    token: String,
    shards: Vec<ReplicaSet>,
}

#[derive(Serialize)]
struct QueryRequest<'a> {
    vector: &'a [f32],
    k: usize,
    filter: Option<&'a Filter>,
    ef_search: usize,
}

impl DistributedCollection {
    pub fn new(shards: Vec<ReplicaSet>, token: impl Into<String>) -> Result<Self> {
        if shards.is_empty() || shards.iter().any(|shard| shard.addresses.is_empty()) {
            return Err(Error::InvalidConfig(
                "distributed collections require at least one replica per shard",
            ));
        }
        let token = token.into();
        if token.is_empty() {
            return Err(Error::InvalidConfig("Raft peer token must not be empty"));
        }
        Ok(Self {
            client: reqwest::Client::new(),
            token,
            shards,
        })
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub async fn upsert(
        &self,
        client_id: impl Into<String>,
        request_id: u64,
        points: Vec<UpsertPoint>,
    ) -> Result<Vec<(usize, ShardResponse)>> {
        if points.is_empty() {
            return Err(Error::InvalidQuery("upsert batch must not be empty"));
        }
        let client_id = client_id.into();
        let mut batches = vec![Vec::new(); self.shards.len()];
        for point in points {
            let shard = shard_index(&point.id, self.shards.len());
            batches[shard].push(point);
        }
        let commands = batches
            .into_iter()
            .enumerate()
            .filter(|(_, points)| !points.is_empty())
            .map(|(shard, points)| {
                (
                    shard,
                    ShardCommand::Upsert {
                        client_id: client_id.clone(),
                        request_id,
                        points,
                    },
                )
            })
            .collect();
        self.write_commands(commands).await
    }

    pub async fn delete(
        &self,
        client_id: impl Into<String>,
        request_id: u64,
        ids: Vec<String>,
    ) -> Result<Vec<(usize, ShardResponse)>> {
        if ids.is_empty() || ids.iter().any(String::is_empty) {
            return Err(Error::InvalidQuery("delete IDs must be non-empty"));
        }
        let client_id = client_id.into();
        let mut batches = vec![Vec::new(); self.shards.len()];
        for id in ids {
            let shard = shard_index(&id, self.shards.len());
            batches[shard].push(id);
        }
        let commands = batches
            .into_iter()
            .enumerate()
            .filter(|(_, ids)| !ids.is_empty())
            .map(|(shard, ids)| {
                (
                    shard,
                    ShardCommand::Delete {
                        client_id: client_id.clone(),
                        request_id,
                        ids,
                    },
                )
            })
            .collect();
        self.write_commands(commands).await
    }

    async fn write_commands(
        &self,
        commands: Vec<(usize, ShardCommand)>,
    ) -> Result<Vec<(usize, ShardResponse)>> {
        let mut tasks = JoinSet::new();
        for (shard, command) in commands {
            let collection = self.clone();
            tasks.spawn(async move {
                let response = collection
                    .send_to_leader(shard, "/v1/raft/write", &command)
                    .await?;
                Ok::<_, Error>((shard, response))
            });
        }
        let mut responses = Vec::new();
        while let Some(result) = tasks.join_next().await {
            responses.push(result.map_err(|error| Error::Concurrency(error.to_string()))??);
        }
        responses.sort_unstable_by_key(|(shard, _)| *shard);
        Ok(responses)
    }

    pub async fn search(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchHit>> {
        self.search_with_ef(vector, k, filter, k.saturating_mul(4).max(96))
            .await
    }

    pub async fn search_with_ef(
        &self,
        vector: Vec<f32>,
        k: usize,
        filter: Option<Filter>,
        ef_search: usize,
    ) -> Result<Vec<SearchHit>> {
        if k == 0 {
            return Err(Error::InvalidQuery("k must be greater than zero"));
        }
        if ef_search < k {
            return Err(Error::InvalidQuery("ef_search must be at least k"));
        }
        let mut tasks = JoinSet::new();
        for shard in 0..self.shards.len() {
            let collection = self.clone();
            let vector = vector.clone();
            let filter = filter.clone();
            tasks.spawn(async move {
                collection
                    .send_to_leader(
                        shard,
                        "/v1/raft/query",
                        &QueryRequest {
                            vector: &vector,
                            k,
                            filter: filter.as_ref(),
                            ef_search,
                        },
                    )
                    .await
            });
        }
        let mut hits = Vec::new();
        while let Some(result) = tasks.join_next().await {
            let shard_hits: Vec<SearchHit> =
                result.map_err(|error| Error::Concurrency(error.to_string()))??;
            hits.extend(shard_hits);
        }
        hits.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    async fn send_to_leader<Req, Response>(
        &self,
        shard: usize,
        path: &str,
        request: &Req,
    ) -> Result<Response>
    where
        Req: Serialize + ?Sized,
        Response: serde::de::DeserializeOwned,
    {
        let mut failures = BTreeMap::new();
        for address in &self.shards[shard].addresses {
            let base = if address.starts_with("http://") || address.starts_with("https://") {
                address.trim_end_matches('/').to_owned()
            } else {
                format!("http://{}", address.trim_end_matches('/'))
            };
            let url = format!("{base}{path}");
            match self
                .client
                .post(&url)
                .header("x-focal-raft-token", &self.token)
                .json(request)
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    return response
                        .json()
                        .await
                        .map_err(|error| Error::Concurrency(error.to_string()));
                }
                Ok(response) => {
                    failures.insert(address.clone(), format!("HTTP {}", response.status()));
                }
                Err(error) => {
                    failures.insert(address.clone(), error.to_string());
                }
            }
        }
        Err(Error::Concurrency(format!(
            "no leader reachable for shard {shard}: {failures:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use openraft::{BasicNode, Config, ServerState};

    use crate::{CollectionConfig, Metric, RaftNode, UpsertPoint, raft_router};

    use super::*;

    fn directory() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "focal-vector-distributed-{}-{nonce}",
            std::process::id()
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn routes_writes_and_merges_global_top_k_across_replicated_shards() {
        let root = directory();
        let config = CollectionConfig {
            dimension: 2,
            metric: Metric::DotProduct,
        };
        let mut nodes = Vec::new();
        let mut servers = Vec::new();
        let mut replica_sets = Vec::new();
        for shard in 0..2 {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap().to_string();
            let node = Arc::new(
                RaftNode::open(
                    1,
                    root.join(format!("shard-{shard}")),
                    config,
                    "secret",
                    Config {
                        cluster_name: format!("shard-{shard}"),
                        heartbeat_interval: 50,
                        election_timeout_min: 150,
                        election_timeout_max: 300,
                        ..Default::default()
                    },
                )
                .await
                .unwrap(),
            );
            servers.push(tokio::spawn({
                let node = Arc::clone(&node);
                async move {
                    axum::serve(listener, raft_router(node)).await.unwrap();
                }
            }));
            node.initialize(BTreeMap::from([(1, BasicNode::new(&address))]))
                .await
                .unwrap();
            node.raft()
                .wait(Some(Duration::from_secs(3)))
                .state(ServerState::Leader, "shard leader")
                .await
                .unwrap();
            nodes.push(node);
            replica_sets.push(ReplicaSet {
                addresses: vec![address],
            });
        }

        let id_for = |wanted| {
            (0..1000)
                .map(|value| format!("point-{value}"))
                .find(|id| shard_index(id, 2) == wanted)
                .unwrap()
        };
        let low = id_for(0);
        let high = id_for(1);
        let collection = DistributedCollection::new(replica_sets, "secret").unwrap();
        let commits = collection
            .upsert(
                "client",
                1,
                vec![
                    UpsertPoint {
                        id: low.clone(),
                        vector: vec![1.0, 0.0],
                        metadata: BTreeMap::new(),
                    },
                    UpsertPoint {
                        id: high.clone(),
                        vector: vec![3.0, 0.0],
                        metadata: BTreeMap::new(),
                    },
                ],
            )
            .await
            .unwrap();
        assert_eq!(commits.len(), 2);
        let hits = collection.search(vec![1.0, 0.0], 2, None).await.unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            [high, low]
        );

        for node in &nodes {
            node.raft().shutdown().await.unwrap();
        }
        for server in servers {
            server.abort();
        }
        drop(nodes);
        fs::remove_dir_all(root).unwrap();
    }
}
