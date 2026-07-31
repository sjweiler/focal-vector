use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimension { expected: usize, actual: usize },
    InvalidVector(&'static str),
    InvalidConfig(&'static str),
    InvalidQuery(&'static str),
    Io(String),
    CorruptStorage(String),
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
            Self::InvalidQuery(message) => write!(f, "invalid query: {message}"),
            Self::Io(message) => write!(f, "storage I/O error: {message}"),
            Self::CorruptStorage(message) => write!(f, "corrupt storage: {message}"),
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
