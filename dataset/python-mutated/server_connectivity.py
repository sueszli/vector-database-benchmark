import socket
from enum import Enum, unique
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from nassl import _nassl
from nassl.ssl_client import ClientCertificateRequested, SslClient
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration
from sslyze.errors import ServerRejectedTlsHandshake, ServerTlsConfigurationNotSupported, TlsHandshakeFailed, ConnectionToServerFailed
from sslyze.connection_helpers.tls_connection import SslConnection, _HANDSHAKE_REJECTED_TLS_ERRORS

@unique
class ClientAuthRequirementEnum(str, Enum):
    """Whether the server asked for client authentication."""
    DISABLED = 'DISABLED'
    OPTIONAL = 'OPTIONAL'
    REQUIRED = 'REQUIRED'

@unique
class TlsVersionEnum(Enum):
    SSL_2_0 = 1
    SSL_3_0 = 2
    TLS_1_0 = 3
    TLS_1_1 = 4
    TLS_1_2 = 5
    TLS_1_3 = 6

@dataclass(frozen=True)
class ServerTlsProbingResult:
    """Additional details about the server, detected via connectivity testing."""
    highest_tls_version_supported: TlsVersionEnum
    cipher_suite_supported: str
    client_auth_requirement: ClientAuthRequirementEnum
    supports_ecdh_key_exchange: bool

def check_connectivity_to_server(server_location: ServerNetworkLocation, network_configuration: ServerNetworkConfiguration) -> ServerTlsProbingResult:
    if False:
        while True:
            i = 10
    'Attempt to perform a full SSL/TLS handshake with the server.\n\n    This method will ensure that the server can be reached, and will also identify one SSL/TLS version and one\n    cipher suite that is supported by the server.\n\n    Args:\n        server_location\n        network_configuration\n\n    Returns:\n        ServerTlsProbingResult\n\n    Raises:\n        ServerConnectivityError: If the server was not reachable or an SSL/TLS handshake could not be completed.\n    '
    tls_detection_result: Optional[_TlsVersionDetectionResult] = None
    try:
        tls_detection_result = _detect_support_for_tls_1_3(server_location=server_location, network_config=network_configuration)
    except _TlsVersionNotSupported:
        pass
    if tls_detection_result is None:
        for tls_version in [TlsVersionEnum.TLS_1_2, TlsVersionEnum.TLS_1_1, TlsVersionEnum.TLS_1_0, TlsVersionEnum.SSL_3_0, TlsVersionEnum.SSL_2_0]:
            try:
                tls_detection_result = _detect_support_for_tls_1_2_or_below(server_location=server_location, network_config=network_configuration, tls_version=tls_version)
                break
            except _TlsVersionNotSupported:
                pass
    if tls_detection_result is None:
        raise ServerTlsConfigurationNotSupported(server_location=server_location, network_configuration=network_configuration, error_message='TLS probing failed: could not find a TLS version and cipher suite supported by the server')
    if tls_detection_result.tls_version_supported == TlsVersionEnum.SSL_2_0:
        raise ServerTlsConfigurationNotSupported(server_location=server_location, network_configuration=network_configuration, error_message="WARNING: Server only supports SSL 2.0 and is therefore affected by critical vulnerabilities. Update the server's software as soon as possible.")
    client_auth_requirement = ClientAuthRequirementEnum.DISABLED
    if tls_detection_result.server_requested_client_cert:
        if tls_detection_result.tls_version_supported.value >= TlsVersionEnum.TLS_1_3.value:
            client_auth_requirement = _detect_client_auth_requirement_with_tls_1_3(server_location=server_location, network_config=network_configuration)
        else:
            client_auth_requirement = _detect_client_auth_requirement_with_tls_1_2_or_below(server_location=server_location, network_config=network_configuration, tls_version=tls_detection_result.tls_version_supported, cipher_list=tls_detection_result.cipher_suite_supported)
    if 'ECDH' in tls_detection_result.cipher_suite_supported:
        is_ecdh_key_exchange_supported = True
    else:
        is_ecdh_key_exchange_supported = _detect_ecdh_support(server_location=server_location, network_config=network_configuration, tls_version=tls_detection_result.tls_version_supported)
    return ServerTlsProbingResult(highest_tls_version_supported=tls_detection_result.tls_version_supported, cipher_suite_supported=tls_detection_result.cipher_suite_supported, client_auth_requirement=client_auth_requirement, supports_ecdh_key_exchange=is_ecdh_key_exchange_supported)

@dataclass(frozen=True)
class ServerConnectivityInfo:
    """All the settings (hostname, port, SSL version, etc.) needed to successfully connect to a given SSL/TLS server.

    Attributes:
        server_location: The minimum information needed to establish a connection to the server.
        network_configuration: Some additional configuration regarding how to connect to the server.
        tls_probing_result: Some additional details about the server's TLS configuration.
    """
    server_location: ServerNetworkLocation
    network_configuration: ServerNetworkConfiguration
    tls_probing_result: ServerTlsProbingResult

    def get_preconfigured_tls_connection(self, override_tls_version: Optional[TlsVersionEnum]=None, ca_certificates_path: Optional[Path]=None, should_use_legacy_openssl: Optional[bool]=None, should_enable_server_name_indication: bool=True) -> SslConnection:
        if False:
            while True:
                i = 10
        'Get an SSLConnection instance with the right SSL configuration for successfully connecting to the server.\n\n        Used by all plugins to connect to the server and run scans.\n        '
        final_ssl_version = self.tls_probing_result.highest_tls_version_supported
        final_openssl_cipher_string: Optional[str]
        final_openssl_cipher_string = self.tls_probing_result.cipher_suite_supported
        if override_tls_version is not None:
            final_ssl_version = override_tls_version
            final_openssl_cipher_string = None
        if should_use_legacy_openssl is not None:
            final_openssl_cipher_string = None
        if self.network_configuration.tls_client_auth_credentials is not None:
            should_ignore_client_auth = False
        else:
            should_ignore_client_auth = True
            if self.tls_probing_result.client_auth_requirement == ClientAuthRequirementEnum.REQUIRED:
                should_ignore_client_auth = False
        ssl_connection = SslConnection(server_location=self.server_location, network_configuration=self.network_configuration, tls_version=final_ssl_version, should_ignore_client_auth=should_ignore_client_auth, ca_certificates_path=ca_certificates_path, should_use_legacy_openssl=should_use_legacy_openssl, should_enable_server_name_indication=should_enable_server_name_indication)
        if final_openssl_cipher_string:
            if final_ssl_version == TlsVersionEnum.TLS_1_3:
                if not isinstance(ssl_connection.ssl_client, SslClient):
                    raise RuntimeError('Should never happen')
                ssl_connection.ssl_client.set_ciphersuites(final_openssl_cipher_string)
            else:
                ssl_connection.ssl_client.set_cipher_list(final_openssl_cipher_string)
        return ssl_connection

@dataclass(frozen=True)
class _TlsVersionDetectionResult:
    tls_version_supported: TlsVersionEnum
    cipher_suite_supported: str
    server_requested_client_cert: bool

class _TlsVersionNotSupported(Exception):
    pass

def _detect_support_for_tls_1_3(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration) -> _TlsVersionDetectionResult:
    if False:
        return 10
    ssl_connection = SslConnection(server_location=server_location, network_configuration=network_config, tls_version=TlsVersionEnum.TLS_1_3, should_ignore_client_auth=False)
    try:
        ssl_connection.connect(should_retry_connection=False)
        return _TlsVersionDetectionResult(tls_version_supported=TlsVersionEnum.TLS_1_3, server_requested_client_cert=False, cipher_suite_supported=ssl_connection.ssl_client.get_current_cipher_name())
    except ClientCertificateRequested:
        return _TlsVersionDetectionResult(tls_version_supported=TlsVersionEnum.TLS_1_3, server_requested_client_cert=True, cipher_suite_supported=ssl_connection.ssl_client.get_current_cipher_name())
    except TlsHandshakeFailed:
        pass
    except (OSError, _nassl.OpenSSLError) as e:
        raise ConnectionToServerFailed(server_location=server_location, network_configuration=network_config, error_message=f'Unexpected connection error: "{e.args}"')
    finally:
        ssl_connection.close()
    raise _TlsVersionNotSupported()

def _detect_support_for_tls_1_2_or_below(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum) -> _TlsVersionDetectionResult:
    if False:
        i = 10
        return i + 15
    if tls_version == TlsVersionEnum.SSL_2_0:
        default_cipher_list = 'SSLv2'
    else:
        default_cipher_list = 'DEFAULT'
    for cipher_list in [default_cipher_list, 'ALL:COMPLEMENTOFALL:-PSK:-SRP']:
        ssl_connection = SslConnection(server_location=server_location, network_configuration=network_config, tls_version=tls_version, should_ignore_client_auth=False)
        ssl_connection.ssl_client.set_cipher_list(cipher_list)
        try:
            ssl_connection.connect(should_retry_connection=False)
            return _TlsVersionDetectionResult(tls_version_supported=tls_version, server_requested_client_cert=False, cipher_suite_supported=ssl_connection.ssl_client.get_current_cipher_name())
        except ClientCertificateRequested:
            return _TlsVersionDetectionResult(tls_version_supported=tls_version, server_requested_client_cert=True, cipher_suite_supported=cipher_list)
        except TlsHandshakeFailed:
            pass
        except (OSError, _nassl.OpenSSLError) as e:
            raise ConnectionToServerFailed(server_location=server_location, network_configuration=network_config, error_message=f'Unexpected connection error: "{e.args}"')
        finally:
            ssl_connection.close()
    raise _TlsVersionNotSupported()

def _detect_client_auth_requirement_with_tls_1_3(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration) -> ClientAuthRequirementEnum:
    if False:
        i = 10
        return i + 15
    'Try to detect if client authentication is optional or required.'
    ssl_connection_auth = SslConnection(server_location=server_location, network_configuration=network_config, tls_version=TlsVersionEnum.TLS_1_3, should_ignore_client_auth=True)
    try:
        ssl_connection_auth.connect(should_retry_connection=False)
        ssl_connection_auth.ssl_client.write(b'A')
        ssl_connection_auth.ssl_client.read(1)
        client_auth_requirement = ClientAuthRequirementEnum.OPTIONAL
    except (ClientCertificateRequested, ServerRejectedTlsHandshake):
        client_auth_requirement = ClientAuthRequirementEnum.REQUIRED
    except socket.timeout:
        client_auth_requirement = ClientAuthRequirementEnum.OPTIONAL
    except _nassl.OpenSSLError as e:
        openssl_error_message = e.args[0]
        is_known_server_rejection_error = False
        for error_msg in _HANDSHAKE_REJECTED_TLS_ERRORS.keys():
            if error_msg in openssl_error_message:
                is_known_server_rejection_error = True
                break
        if is_known_server_rejection_error:
            client_auth_requirement = ClientAuthRequirementEnum.REQUIRED
        else:
            raise
    except OSError as e:
        raise ConnectionToServerFailed(server_location=server_location, network_configuration=network_config, error_message=f'Unexpected connection error: "{e.args}"')
    finally:
        ssl_connection_auth.close()
    return client_auth_requirement

def _detect_client_auth_requirement_with_tls_1_2_or_below(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum, cipher_list: str) -> ClientAuthRequirementEnum:
    if False:
        i = 10
        return i + 15
    'Try to detect if client authentication is optional or required.'
    if tls_version.value >= TlsVersionEnum.TLS_1_3.value:
        raise ValueError('Use _detect_client_auth_requirement_with_tls_1_3()')
    ssl_connection_auth = SslConnection(server_location=server_location, network_configuration=network_config, tls_version=tls_version, should_ignore_client_auth=True)
    ssl_connection_auth.ssl_client.set_cipher_list(cipher_list)
    try:
        ssl_connection_auth.connect(should_retry_connection=False)
        client_auth_requirement = ClientAuthRequirementEnum.OPTIONAL
    except (ClientCertificateRequested, ServerRejectedTlsHandshake):
        client_auth_requirement = ClientAuthRequirementEnum.REQUIRED
    finally:
        ssl_connection_auth.close()
    return client_auth_requirement

def _detect_ecdh_support(server_location: ServerNetworkLocation, network_config: ServerNetworkConfiguration, tls_version: TlsVersionEnum) -> bool:
    if False:
        i = 10
        return i + 15
    if tls_version.value < TlsVersionEnum.TLS_1_2.value:
        return False
    is_ecdh_key_exchange_supported = False
    ssl_connection = SslConnection(server_location=server_location, network_configuration=network_config, tls_version=tls_version, should_use_legacy_openssl=False, should_ignore_client_auth=True)
    if not isinstance(ssl_connection.ssl_client, SslClient):
        raise RuntimeError("Should never happen: specified should_use_legacy_openssl=False but didn't get the modern SSL client")
    enable_ecdh_cipher_suites(tls_version, ssl_connection.ssl_client)
    try:
        ssl_connection.connect(should_retry_connection=False)
        is_ecdh_key_exchange_supported = True
    except ClientCertificateRequested:
        is_ecdh_key_exchange_supported = True
    except ServerRejectedTlsHandshake:
        is_ecdh_key_exchange_supported = False
    finally:
        ssl_connection.close()
    return is_ecdh_key_exchange_supported

def enable_ecdh_cipher_suites(tls_version: TlsVersionEnum, ssl_client: SslClient) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Set the elliptic curve cipher suites.'
    if tls_version == TlsVersionEnum.TLS_1_3:
        ssl_client.set_ciphersuites('TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_CCM_SHA256:TLS_AES_128_CCM_8_SHA256')
    else:
        ssl_client.set_cipher_list('ECDH')