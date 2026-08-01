//! Blocking local client for the Focal Vector sidecar.
//!
//! The client intentionally depends only on the protocol crate and JSON. Async
//! services should call it through their runtime's blocking-task facility.

use std::fmt;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub use focal_vector_protocol::{
    CollectionInfo, Filter, Metric, Point, Request, Response, SearchHit,
};
use focal_vector_protocol::{DEFAULT_SOCKET_NAME, Envelope, PROTOCOL_VERSION, SOCKET_ENV};

const DEFAULT_MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    Configuration(String),
    Io(std::io::Error),
    Protocol(String),
    Service { code: String, message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) | Self::Protocol(message) => formatter.write_str(message),
            Self::Io(error) => write!(formatter, "Focal Vector IPC failed: {error}"),
            Self::Service { code, message } => write!(formatter, "Focal Vector {code}: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Client {
    socket: PathBuf,
    timeout: Duration,
    max_response_bytes: u64,
}

impl Client {
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
            timeout: Duration::from_secs(30),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    pub fn from_environment() -> Result<Self> {
        Ok(Self::new(socket_path()?))
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn hello(&self) -> Result<String> {
        match self.request(&Request::Hello)? {
            Response::Hello { server_version } => Ok(server_version),
            response => Err(unexpected("hello", response)),
        }
    }

    pub fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        match self.request(&Request::ListCollections)? {
            Response::Collections { collections } => Ok(collections),
            response => Err(unexpected("list_collections", response)),
        }
    }

    pub fn create_collection(
        &self,
        name: impl Into<String>,
        dimension: usize,
        metric: Metric,
    ) -> Result<CollectionInfo> {
        match self.request(&Request::CreateCollection {
            name: name.into(),
            dimension,
            metric,
        })? {
            Response::Created { collection } => Ok(collection),
            response => Err(unexpected("create_collection", response)),
        }
    }

    pub fn upsert(&self, collection: impl Into<String>, points: Vec<Point>) -> Result<u64> {
        self.sequence(Request::Upsert {
            collection: collection.into(),
            points,
        })
    }

    pub fn delete(&self, collection: impl Into<String>, ids: Vec<String>) -> Result<u64> {
        self.sequence(Request::Delete {
            collection: collection.into(),
            ids,
        })
    }

    pub fn query(
        &self,
        collection: impl Into<String>,
        vector: Vec<f32>,
        k: usize,
        filter: Option<Filter>,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchHit>> {
        match self.request(&Request::Query {
            collection: collection.into(),
            vector,
            k,
            filter,
            ef_search,
        })? {
            Response::Query { hits } => Ok(hits),
            response => Err(unexpected("query", response)),
        }
    }

    pub fn flush(&self, collection: impl Into<String>) -> Result<u64> {
        self.sequence(Request::Flush {
            collection: collection.into(),
        })
    }

    pub fn request(&self, request: &Request) -> Result<Response> {
        let mut stream = UnixStream::connect(&self.socket)?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        let body = serde_json::to_vec(&Envelope::current(request))
            .map_err(|error| Error::Protocol(format!("could not encode IPC request: {error}")))?;
        stream.write_all(&body)?;
        stream.shutdown(std::net::Shutdown::Write)?;

        let mut response = Vec::new();
        stream
            .take(self.max_response_bytes + 1)
            .read_to_end(&mut response)?;
        if response.len() as u64 > self.max_response_bytes {
            return Err(Error::Protocol(
                "IPC response exceeds the configured limit".into(),
            ));
        }
        let envelope: Envelope<Response> = serde_json::from_slice(&response)
            .map_err(|error| Error::Protocol(format!("invalid IPC response: {error}")))?;
        if envelope.protocol_version != PROTOCOL_VERSION {
            return Err(Error::Protocol(format!(
                "unsupported IPC protocol version {}; client supports {}",
                envelope.protocol_version, PROTOCOL_VERSION
            )));
        }
        match envelope.payload {
            Response::Error { code, message } => Err(Error::Service { code, message }),
            response => Ok(response),
        }
    }

    fn sequence(&self, request: Request) -> Result<u64> {
        match self.request(&request)? {
            Response::Sequence { sequence } => Ok(sequence),
            response => Err(unexpected("mutation", response)),
        }
    }
}

pub fn socket_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os(SOCKET_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Error::Configuration(
                "XDG_RUNTIME_DIR is required when FOCAL_VECTOR_SOCKET is unset".into(),
            )
        })?;
    Ok(PathBuf::from(runtime)
        .join("focaldesk")
        .join(DEFAULT_SOCKET_NAME))
}

fn unexpected(operation: &str, response: Response) -> Error {
    Error::Protocol(format!("unexpected {operation} response: {response:?}"))
}
