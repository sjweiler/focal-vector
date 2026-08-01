use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimension { expected: usize, actual: usize },
    InvalidVector(&'static str),
    InvalidConfig(&'static str),
    InvalidConfiguration(String),
    InvalidQuery(&'static str),
    Io(String),
    CorruptStorage(String),
    Concurrency(String),
    NotFound(String),
    AlreadyExists(String),
    Unauthorized,
    ResourceExhausted(String),
    SequenceOverflow,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension { expected, actual } => {
                write!(
                    f,
                    "invalid vector dimension: expected {expected}, got {actual}"
                )
            }
            Self::InvalidVector(message) => write!(f, "invalid vector: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid collection config: {message}"),
            Self::InvalidConfiguration(message) => write!(f, "invalid configuration: {message}"),
            Self::InvalidQuery(message) => write!(f, "invalid query: {message}"),
            Self::Io(message) => write!(f, "storage I/O error: {message}"),
            Self::CorruptStorage(message) => write!(f, "corrupt storage: {message}"),
            Self::Concurrency(message) => write!(f, "concurrency error: {message}"),
            Self::NotFound(message) => write!(f, "not found: {message}"),
            Self::AlreadyExists(message) => write!(f, "already exists: {message}"),
            Self::Unauthorized => f.write_str("unauthorized"),
            Self::ResourceExhausted(message) => write!(f, "resource exhausted: {message}"),
            Self::SequenceOverflow => f.write_str("collection sequence overflow"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}
