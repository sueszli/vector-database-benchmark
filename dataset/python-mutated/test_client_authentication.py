import pytest
from sslyze.json.json_output import _ServerTlsProbingResultAsJson
from sslyze.server_connectivity import ClientAuthRequirementEnum, check_connectivity_to_server
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer, ClientAuthConfigEnum, LegacyOpenSslServer

class TestClientAuthentication:

    def test_optional_client_authentication(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation(hostname='client.badssl.com', port=443)
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.cipher_suite_supported
        assert tls_probing_result.client_auth_requirement == ClientAuthRequirementEnum.OPTIONAL
        server_info_as_json = _ServerTlsProbingResultAsJson.from_orm(tls_probing_result)
        assert server_info_as_json.json()

@can_only_run_on_linux_64
class TestClientAuthenticationWithLocalServer:

    def test_optional_client_auth(self):
        if False:
            i = 10
            return i + 15
        with ModernOpenSslServer(client_auth_config=ClientAuthConfigEnum.OPTIONAL) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result.client_auth_requirement == ClientAuthRequirementEnum.OPTIONAL

    def test_required_client_auth_tls_1_2(self):
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result.client_auth_requirement == ClientAuthRequirementEnum.REQUIRED

    @pytest.mark.skip(reason='Client auth config detection with TLS 1.3 is broken; fix me')
    def test_required_client_auth_tls_1_3(self):
        if False:
            while True:
                i = 10
        with ModernOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, port=server.port, ip_address=server.ip_address)
            tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        assert tls_probing_result.client_auth_requirement == ClientAuthRequirementEnum.REQUIRED