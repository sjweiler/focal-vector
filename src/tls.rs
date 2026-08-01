use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum_server::tls_rustls::RustlsConfig;
use rustls::server::WebPkiClientVerifier;
use rustls::{RootCertStore, ServerConfig};
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpClientTlsConfig {
    /// Additional private CA. Omit when the server chains to a public web PKI root.
    pub ca_certificate: Option<PathBuf>,
    /// PEM containing the client certificate chain and its private key.
    pub identity_pem: Option<PathBuf>,
}

pub fn build_http_client(tls: Option<&HttpClientTlsConfig>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder();
    if let Some(tls) = tls {
        if let Some(ca_certificate) = &tls.ca_certificate {
            let ca = std::fs::read(ca_certificate)?;
            let certificate = reqwest::Certificate::from_pem(&ca)
                .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
            builder = builder.add_root_certificate(certificate);
        }
        if let Some(identity_path) = &tls.identity_pem {
            let identity = reqwest::Identity::from_pem(&std::fs::read(identity_path)?)
                .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
            builder = builder.identity(identity);
        }
    }
    builder
        .build()
        .map_err(|error| Error::InvalidConfiguration(error.to_string()))
}

pub fn load_server_tls(
    certificate_path: impl AsRef<Path>,
    private_key_path: impl AsRef<Path>,
    client_ca_path: Option<&Path>,
) -> Result<RustlsConfig> {
    let certificates = CertificateDer::pem_file_iter(certificate_path)
        .map_err(|error| Error::InvalidConfiguration(error.to_string()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
    if certificates.is_empty() {
        return Err(Error::InvalidConfiguration(
            "TLS certificate file contains no certificates".into(),
        ));
    }
    let private_key = PrivateKeyDer::from_pem_file(private_key_path)
        .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;

    let builder = ServerConfig::builder();
    let mut config = if let Some(client_ca_path) = client_ca_path {
        let client_roots = CertificateDer::pem_file_iter(client_ca_path)
            .map_err(|error| Error::InvalidConfiguration(error.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
        let mut roots = RootCertStore::empty();
        let (accepted, rejected) = roots.add_parsable_certificates(client_roots);
        if accepted == 0 || rejected > 0 {
            return Err(Error::InvalidConfiguration(format!(
                "client CA file contained {accepted} accepted and {rejected} rejected certificates"
            )));
        }
        let verifier = WebPkiClientVerifier::builder(Arc::new(roots))
            .build()
            .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
        builder
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, private_key)
    } else {
        builder
            .with_no_client_auth()
            .with_single_cert(certificates, private_key)
    }
    .map_err(|error| Error::InvalidConfiguration(error.to_string()))?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(RustlsConfig::from_config(Arc::new(config)))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use rcgen::{CertifiedKey, generate_simple_self_signed};

    use super::*;

    #[test]
    fn loads_native_tls_mtls_and_client_identity() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("focal-vector-tls-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let CertifiedKey { cert, signing_key } =
            generate_simple_self_signed(["localhost".into()]).unwrap();
        let certificate = directory.join("certificate.pem");
        let private_key = directory.join("private-key.pem");
        let identity = directory.join("identity.pem");
        fs::write(&certificate, cert.pem()).unwrap();
        fs::write(&private_key, signing_key.serialize_pem()).unwrap();
        fs::write(
            &identity,
            format!("{}{}", cert.pem(), signing_key.serialize_pem()),
        )
        .unwrap();

        load_server_tls(&certificate, &private_key, None).unwrap();
        load_server_tls(&certificate, &private_key, Some(&certificate)).unwrap();
        build_http_client(Some(&HttpClientTlsConfig {
            ca_certificate: Some(certificate),
            identity_pem: Some(identity),
        }))
        .unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
}
