import pytest
from sslyze.plugins.compression_plugin import CompressionImplementation, CompressionScanResult
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import LegacyOpenSslServer, ClientAuthConfigEnum

class TestCompressionPlugin:

    def test_compression_disabled(self) -> None:
        if False:
            return 10
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: CompressionScanResult = CompressionImplementation.scan_server(server_info)
        assert not result.supports_compression
        assert CompressionImplementation.cli_connector_cls.result_to_console_output(result)

    @pytest.mark.skip('Not implemented; find a server vulnerable to TLS compression')
    def test_compression_enabled(self) -> None:
        if False:
            while True:
                i = 10
        pass

    @can_only_run_on_linux_64
    def test_succeeds_when_client_auth_failed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: CompressionScanResult = CompressionImplementation.scan_server(server_info)
        assert not result.supports_compression