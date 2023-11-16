from sslyze.plugins.openssl_ccs_injection_plugin import OpenSslCcsInjectionImplementation
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import LegacyOpenSslServer, ClientAuthConfigEnum

class TestOpenSslCcsInjectionPlugin:

    def test_not_vulnerable(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation('www.google.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result = OpenSslCcsInjectionImplementation.scan_server(server_info)
        assert not result.is_vulnerable_to_ccs_injection
        assert OpenSslCcsInjectionImplementation.cli_connector_cls.result_to_console_output(result)

    def test_not_vulnerable_and_server_has_cloudfront_bug(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation(hostname='uol.com', port=443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result = OpenSslCcsInjectionImplementation.scan_server(server_info)
        assert not result.is_vulnerable_to_ccs_injection

    @can_only_run_on_linux_64
    def test_vulnerable(self):
        if False:
            i = 10
            return i + 15
        with LegacyOpenSslServer() as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result = OpenSslCcsInjectionImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_ccs_injection
        assert OpenSslCcsInjectionImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_vulnerable_and_server_has_sni_bug(self):
        if False:
            print('Hello World!')
        server_name_indication = 'server.com'
        with LegacyOpenSslServer(require_server_name_indication_value=server_name_indication) as server:
            server_location = ServerNetworkLocation(hostname=server_name_indication, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            object.__setattr__(server_info.network_configuration, 'tls_server_name_indication', 'wrongvalue.com')
            result = OpenSslCcsInjectionImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_ccs_injection

    @can_only_run_on_linux_64
    def test_succeeds_when_client_auth_failed(self):
        if False:
            print('Hello World!')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result = OpenSslCcsInjectionImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_ccs_injection