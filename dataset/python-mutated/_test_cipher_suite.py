from dataclasses import dataclass
from typing import Optional, Union
from nassl.ephemeral_key_info import EphemeralKeyInfo
from nassl.legacy_ssl_client import LegacySslClient
from nassl.ssl_client import ClientCertificateRequested, SslClient, BaseSslClient
from sslyze.errors import ServerRejectedTlsHandshake, ServerTlsConfigurationNotSupported, TlsHandshakeTimedOut
from sslyze.plugins.openssl_cipher_suites.cipher_suites import CipherSuite
from sslyze.server_connectivity import ServerConnectivityInfo, TlsVersionEnum
from sslyze.plugins.openssl_cipher_suites._tls12_workaround import WorkaroundForTls12ForCipherSuites

@dataclass(frozen=True)
class CipherSuiteAcceptedByServer:
    """
    ephemeral_key: The ephemeral key negotiated with the server when using (EC) DH cipher suites. None if the cipher
        suite does not use ephemeral keys or if the ephemeral key could not be retrieved.
    """
    cipher_suite: CipherSuite
    ephemeral_key: Optional[EphemeralKeyInfo]

@dataclass(frozen=True)
class CipherSuiteRejectedByServer:
    cipher_suite: CipherSuite
    error_message: str

def connect_with_cipher_suite(server_connectivity_info: ServerConnectivityInfo, tls_version: TlsVersionEnum, cipher_suite: CipherSuite) -> Union[CipherSuiteAcceptedByServer, CipherSuiteRejectedByServer]:
    if False:
        return 10
    'Initiates a SSL handshake with the server using the SSL version and the cipher suite specified.'
    requires_legacy_openssl = True
    if tls_version == TlsVersionEnum.TLS_1_2:
        requires_legacy_openssl = WorkaroundForTls12ForCipherSuites.requires_legacy_openssl(cipher_suite.openssl_name)
    elif tls_version == TlsVersionEnum.TLS_1_3:
        requires_legacy_openssl = False
    ssl_connection = server_connectivity_info.get_preconfigured_tls_connection(override_tls_version=tls_version, should_use_legacy_openssl=requires_legacy_openssl)
    _set_cipher_suite_string(tls_version, cipher_suite.openssl_name, ssl_connection.ssl_client)
    ephemeral_key = None
    try:
        ssl_connection.connect()
        ephemeral_key = ssl_connection.ssl_client.get_ephemeral_key()
    except ServerTlsConfigurationNotSupported:
        pass
    except ClientCertificateRequested:
        ephemeral_key = ssl_connection.ssl_client.get_ephemeral_key()
        pass
    except ServerRejectedTlsHandshake as e:
        return CipherSuiteRejectedByServer(cipher_suite=cipher_suite, error_message=e.error_message)
    except TlsHandshakeTimedOut as e:
        return CipherSuiteRejectedByServer(cipher_suite=cipher_suite, error_message=e.error_message)
    finally:
        ssl_connection.close()
    return CipherSuiteAcceptedByServer(cipher_suite=cipher_suite, ephemeral_key=ephemeral_key)

def _set_cipher_suite_string(tls_version: TlsVersionEnum, cipher_suite_str: str, ssl_client: BaseSslClient) -> None:
    if False:
        print('Hello World!')
    if isinstance(ssl_client, SslClient):
        if tls_version == TlsVersionEnum.TLS_1_3:
            ssl_client.set_ciphersuites(cipher_suite_str)
        else:
            ssl_client.set_cipher_list(cipher_suite_str)
    elif isinstance(ssl_client, LegacySslClient):
        ssl_client.set_cipher_list(cipher_suite_str)
    else:
        raise RuntimeError('Should never happen')