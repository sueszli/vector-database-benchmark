import socket
import pytest
from sslyze.json.json_output import _ServerTlsProbingResultAsJson
from tests.openssl_server import LegacyOpenSslServer
from sslyze.server_connectivity import TlsVersionEnum, check_connectivity_to_server
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration
from sslyze.errors import ConnectionToServerTimedOut, ServerRejectedConnection, ServerTlsConfigurationNotSupported, ConnectionToServerFailed
from tests.markers import can_only_run_on_linux_64

def _is_ipv6_available() -> bool:
    if False:
        for i in range(10):
            print('nop')
    has_ipv6 = False
    s = socket.socket(socket.AF_INET6)
    try:
        s.connect(('2607:f8b0:4005:804::2004', 443))
        has_ipv6 = True
    except Exception:
        pass
    finally:
        s.close()
    return has_ipv6

class TestServerConnectivityTester:

    def test_via_direct_connection(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation('www.google.com', 443)
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result.cipher_suite_supported
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.client_auth_requirement
        assert tls_probing_result.supports_ecdh_key_exchange
        tls_probing_result_as_json = _ServerTlsProbingResultAsJson.from_orm(tls_probing_result)
        assert tls_probing_result_as_json.json()

    def test_via_direct_connection_but_server_timed_out(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation(hostname='notarealdomain.not.real.notreal.not', port=1234, ip_address='123.123.123.123')
        with pytest.raises(ConnectionToServerTimedOut):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_via_direct_connection_but_server_rejected_connection(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation(hostname='localhost', port=1234)
        with pytest.raises(ServerRejectedConnection):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_via_direct_connection_but_server_tls_config_not_supported(self):
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation(hostname='dh480.badssl.com', port=443)
        with pytest.raises(ServerTlsConfigurationNotSupported):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_tls_1_only(self):
        if False:
            return 10
        server_location = ServerNetworkLocation(hostname='tls-v1-0.badssl.com', port=1010)
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result
        assert tls_probing_result.client_auth_requirement
        assert tls_probing_result.cipher_suite_supported
        assert tls_probing_result.highest_tls_version_supported == TlsVersionEnum.TLS_1_0

    @pytest.mark.skipif(not _is_ipv6_available(), reason='IPv6 not available')
    def test_ipv6(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, ip_address='2607:f8b0:4005:804::2004')
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result
        assert tls_probing_result.client_auth_requirement
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.cipher_suite_supported

    def test_international_hostname(self):
        if False:
            return 10
        server_location = ServerNetworkLocation(hostname='www.société.com', port=443)
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result
        assert tls_probing_result.client_auth_requirement
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.cipher_suite_supported
        tls_probing_result_as_json = _ServerTlsProbingResultAsJson.from_orm(tls_probing_result)
        assert tls_probing_result_as_json.json()

    @can_only_run_on_linux_64
    def test_server_triggers_unexpected_connection_error(self):
        if False:
            print('Hello World!')
        with LegacyOpenSslServer(require_server_name_indication_value='server.com') as server:
            server_location = ServerNetworkLocation(hostname='not_the_right_value.com', ip_address=server.ip_address, port=server.port)
            with pytest.raises(ConnectionToServerFailed, match='unrecognized name'):
                check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    @can_only_run_on_linux_64
    def test_server_only_supports_sslv2(self):
        if False:
            while True:
                i = 10
        with LegacyOpenSslServer(openssl_cipher_string='SSLv2') as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            with pytest.raises(ConnectionToServerFailed, match='SSL 2.0'):
                check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))