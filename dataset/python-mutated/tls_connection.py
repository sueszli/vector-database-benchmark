import socket
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from nassl.legacy_ssl_client import LegacySslClient
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration, ConnectionTypeEnum
from sslyze.errors import ConnectionToServerTimedOut, ServerRejectedConnection, ConnectionToServerFailed, ConnectionToHttpProxyTimedOut, HttpProxyRejectedConnection, ConnectionToHttpProxyFailed, ServerRejectedOpportunisticTlsNegotiation, ServerRejectedTlsHandshake, ServerTlsConfigurationNotSupported, TlsHandshakeTimedOut
from sslyze.connection_helpers.http_response_parser import HttpResponseParser
import time
from nassl import _nassl
from nassl.ssl_client import SslClient, OpenSslVersionEnum, BaseSslClient, OpenSslVerifyEnum
from nassl.ssl_client import ClientCertificateRequested
from sslyze.connection_helpers.opportunistic_tls_helpers import get_opportunistic_tls_helper, OpportunisticTlsError
if TYPE_CHECKING:
    from sslyze.server_connectivity import TlsVersionEnum

def _open_socket_for_direct_connection(server_location: ServerNetworkLocation, network_timeout: int) -> socket.socket:
    if False:
        return 10
    assert server_location.ip_address
    return socket.create_connection((server_location.ip_address, server_location.port), timeout=network_timeout)

class _ConnectionToHttpProxyTimedOut(Exception):
    pass

class _HttpProxyRejectedConnection(Exception):
    pass

class _ConnectionToHttpProxyFailed(Exception):
    pass

def _open_socket_for_connection_via_http_proxy(server_location: ServerNetworkLocation, network_timeout: int) -> socket.socket:
    if False:
        while True:
            i = 10
    assert server_location.http_proxy_settings
    try:
        sock = socket.create_connection((server_location.http_proxy_settings.hostname, server_location.http_proxy_settings.port), timeout=network_timeout)
        proxy_authorization_header = server_location.http_proxy_settings.proxy_authorization_header
        if proxy_authorization_header is None:
            sock.send(f'CONNECT {server_location.hostname}:{server_location.port} HTTP/1.1\r\n\r\n'.encode('utf-8'))
        else:
            sock.send(f'CONNECT {server_location.hostname}:{server_location.port} HTTP/1.1\r\nProxy-Authorization: Basic {proxy_authorization_header}\r\n\r\n'.encode('utf-8'))
        http_response = HttpResponseParser.parse_from_socket(sock)
    except socket.timeout:
        raise _ConnectionToHttpProxyTimedOut()
    except ConnectionError:
        raise _HttpProxyRejectedConnection('The HTTP proxy rejected the connection')
    except socket.error:
        raise _ConnectionToHttpProxyFailed()
    if http_response.status != 200:
        raise _HttpProxyRejectedConnection('The HTTP proxy rejected the CONNECT request')
    return sock

def _open_socket(server_location: ServerNetworkLocation, network_timeout: int) -> socket.socket:
    if False:
        while True:
            i = 10
    if server_location.connection_type == ConnectionTypeEnum.VIA_HTTP_PROXY:
        return _open_socket_for_connection_via_http_proxy(server_location, network_timeout)
    elif server_location.connection_type == ConnectionTypeEnum.DIRECT:
        return _open_socket_for_direct_connection(server_location, network_timeout)
    else:
        raise ValueError()
_HANDSHAKE_REJECTED_TLS_ERRORS = {'excessive message size': 'TLS error: excessive message size', 'bad mac decode': 'TLS error: bad mac decode', 'wrong version number': 'TLS error: wrong version number', 'no cipher match': 'TLS error: no cipher match', 'bad decompression': 'TLS error: bad decompression', 'peer error no cipher': 'TLS error: peer error no cipher', 'no cipher list': 'TLS error: no ciphers list', 'insufficient security': 'TLS error: insufficient security', 'block type is not 01': 'TLS error: block type is not 01', 'wrong ssl version': 'TLS error: wrong SSL version', 'digest check failed': 'TLS error: digest check failed', 'sslv3 alert handshake failure': 'TLS alert: handshake failure', 'tlsv1 alert protocol version': 'TLS alert: protocol version ', 'tlsv1 alert decrypt error': 'TLS alert: Decrypt error', 'tlsv1 alert decode error': 'TLS alert: Decode error', 'Connection was shut down by peer': 'Server closed the connection during the TLS handshake', 'bad record mac': 'TLS alert: bad record mac', 'tlsv1 alert internal error': 'TLS alert: Internal error', 'illegal padding': 'TLS alert: Illegal padding', 'illegal parameter': 'TLS alert: Illegal parameter', 'wrong certificate type': 'Server returned wrong certificate type'}

class NoCiphersAvailableBugInSSlyze(Exception):
    """Should never happen."""

class SslConnection:
    """SSL connection that handles error processing, including retries when receiving timeouts.

    This it the base class to use to connect to a server in order to scan it.
    """

    def __init__(self, server_location: ServerNetworkLocation, network_configuration: ServerNetworkConfiguration, tls_version: 'TlsVersionEnum', should_ignore_client_auth: bool, should_use_legacy_openssl: Optional[bool]=None, ca_certificates_path: Optional[Path]=None, should_enable_server_name_indication: bool=True) -> None:
        if False:
            return 10
        self._server_location = server_location
        self._network_configuration = network_configuration
        nassl_tls_version = OpenSslVersionEnum(tls_version.value)
        self.ssl_client: BaseSslClient
        final_should_use_legacy_openssl: bool
        if should_use_legacy_openssl is None:
            final_should_use_legacy_openssl = False if nassl_tls_version in [OpenSslVersionEnum.TLSV1_2, OpenSslVersionEnum.TLSV1_3] else True
        else:
            final_should_use_legacy_openssl = should_use_legacy_openssl
        if nassl_tls_version == OpenSslVersionEnum.TLSV1_3 and final_should_use_legacy_openssl:
            raise ValueError('Cannot use legacy OpenSSL with TLS 1.3')
        elif nassl_tls_version in [OpenSslVersionEnum.SSLV2, OpenSslVersionEnum.SSLV3] and (not final_should_use_legacy_openssl):
            raise ValueError('Cannot use modern OpenSSL with SSL 2.0 or 3.0')
        ssl_client_cls = LegacySslClient if final_should_use_legacy_openssl else SslClient
        if network_configuration.tls_client_auth_credentials:
            self.ssl_client = ssl_client_cls(ssl_version=nassl_tls_version, ssl_verify=OpenSslVerifyEnum.NONE, ssl_verify_locations=ca_certificates_path, client_certificate_chain=network_configuration.tls_client_auth_credentials.certificate_chain_path, client_key=network_configuration.tls_client_auth_credentials.key_path, client_key_type=network_configuration.tls_client_auth_credentials.key_type, client_key_password=network_configuration.tls_client_auth_credentials.key_password, ignore_client_authentication_requests=False)
        else:
            self.ssl_client = ssl_client_cls(ssl_version=nassl_tls_version, ssl_verify=OpenSslVerifyEnum.NONE, ssl_verify_locations=ca_certificates_path, ignore_client_authentication_requests=should_ignore_client_auth)
        if nassl_tls_version != OpenSslVersionEnum.TLSV1_3:
            self.ssl_client.set_cipher_list('HIGH:MEDIUM:-aNULL:-eNULL:-3DES:-SRP:-PSK:-CAMELLIA')
        if should_enable_server_name_indication and nassl_tls_version != OpenSslVersionEnum.SSLV2:
            self.ssl_client.set_tlsext_host_name(network_configuration.tls_server_name_indication)

    def _do_pre_handshake(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            sock = _open_socket(self._server_location, self._network_configuration.network_timeout)
        except _ConnectionToHttpProxyTimedOut:
            raise ConnectionToHttpProxyTimedOut(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Connection to HTTP Proxy timed out')
        except _HttpProxyRejectedConnection as e:
            raise HttpProxyRejectedConnection(server_location=self._server_location, network_configuration=self._network_configuration, error_message=e.args[0])
        except _ConnectionToHttpProxyFailed:
            raise ConnectionToHttpProxyFailed(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Connection to the HTTP proxy failed')
        if self._network_configuration.tls_opportunistic_encryption:
            opportunistic_tls_helper = get_opportunistic_tls_helper(self._network_configuration.tls_opportunistic_encryption, self._network_configuration.xmpp_to_hostname)
            try:
                opportunistic_tls_helper.prepare_socket_for_tls_handshake(sock)
            except OpportunisticTlsError as e:
                raise ServerRejectedOpportunisticTlsNegotiation(server_location=self._server_location, error_message=e.args[0], network_configuration=self._network_configuration)
        self.ssl_client.set_underlying_socket(sock)

    def connect(self, should_retry_connection: bool=True) -> None:
        if False:
            while True:
                i = 10
        max_attempts_nb = self._network_configuration.network_max_retries if should_retry_connection else 1
        connection_attempts_nb = 0
        delay_for_next_attempt = 0
        while True:
            time.sleep(delay_for_next_attempt)
            try:
                self._do_pre_handshake()
            except socket.timeout:
                connection_attempts_nb += 1
                if connection_attempts_nb >= max_attempts_nb:
                    raise ConnectionToServerTimedOut(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Connection to the server timed out')
                elif connection_attempts_nb == 1:
                    delay_for_next_attempt = 1
                else:
                    delay_for_next_attempt = min(6, 2 * delay_for_next_attempt)
            except ConnectionError:
                raise ServerRejectedConnection(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Server rejected the connection')
            except OSError:
                raise ConnectionToServerFailed(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Connection to the server failed')
            else:
                break
        try:
            self.ssl_client.do_handshake()
        except ClientCertificateRequested:
            raise
        except socket.timeout:
            raise TlsHandshakeTimedOut(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Connection to server timed out during the TLS handshake')
        except ConnectionError:
            raise ServerRejectedTlsHandshake(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Server rejected the connection')
        except OSError as e:
            if 'Nassl SSL handshake failed' in e.args[0]:
                raise ServerRejectedTlsHandshake(server_location=self._server_location, network_configuration=self._network_configuration, error_message='Server interrupted the TLS handshake')
            raise
        except _nassl.OpenSSLError as e:
            openssl_error_message = e.args[0]
            if 'dh key too small' in openssl_error_message:
                raise ServerTlsConfigurationNotSupported(server_location=self._server_location, network_configuration=self._network_configuration, error_message='DH key too small')
            if 'no ciphers available' in openssl_error_message:
                raise NoCiphersAvailableBugInSSlyze(f'Set a cipher that is not supported by nassl: {self.ssl_client.get_cipher_list()}')
            for error_msg in _HANDSHAKE_REJECTED_TLS_ERRORS.keys():
                if error_msg in openssl_error_message:
                    raise ServerRejectedTlsHandshake(server_location=self._server_location, network_configuration=self._network_configuration, error_message=_HANDSHAKE_REJECTED_TLS_ERRORS[error_msg])
            raise

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.ssl_client.shutdown()