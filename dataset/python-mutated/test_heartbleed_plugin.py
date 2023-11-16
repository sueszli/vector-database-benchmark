from sslyze.plugins.heartbleed_plugin import HeartbleedImplementation
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import LegacyOpenSslServer, ClientAuthConfigEnum

class TestHeartbleedPlugin:

    def test_not_vulnerable(self):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.google.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result = HeartbleedImplementation.scan_server(server_info)
        assert not result.is_vulnerable_to_heartbleed
        assert HeartbleedImplementation.cli_connector_cls.result_to_console_output(result)

    def test_not_vulnerable_and_server_has_cloudfront_bug(self):
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation(hostname='uol.com', port=443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result = HeartbleedImplementation.scan_server(server_info)
        assert not result.is_vulnerable_to_heartbleed

    @can_only_run_on_linux_64
    def test_vulnerable(self):
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer() as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result = HeartbleedImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_heartbleed
        assert HeartbleedImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_vulnerable_and_server_has_sni_bug(self):
        if False:
            i = 10
            return i + 15
        server_name_indication = 'server.com'
        with LegacyOpenSslServer(require_server_name_indication_value=server_name_indication) as server:
            server_location = ServerNetworkLocation(hostname=server_name_indication, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            object.__setattr__(server_info.network_configuration, 'tls_server_name_indication', 'wrongvalue.com')
            result = HeartbleedImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_heartbleed

    @can_only_run_on_linux_64
    def test_succeeds_when_client_auth_failed(self):
        if False:
            return 10
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result = HeartbleedImplementation.scan_server(server_info)
        assert result.is_vulnerable_to_heartbleed