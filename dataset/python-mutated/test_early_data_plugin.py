from sslyze.plugins.early_data_plugin import EarlyDataScanResult, EarlyDataImplementation
from sslyze.server_setting import ServerNetworkLocation
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer, LegacyOpenSslServer

class TestEarlyDataPlugin:

    @can_only_run_on_linux_64
    def test_early_data_enabled(self) -> None:
        if False:
            while True:
                i = 10
        with ModernOpenSslServer(max_early_data=256) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: EarlyDataScanResult = EarlyDataImplementation.scan_server(server_info)
        assert result.supports_early_data
        assert EarlyDataImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_early_data_disabled_no_tls_1_3(self) -> None:
        if False:
            return 10
        with LegacyOpenSslServer() as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: EarlyDataScanResult = EarlyDataImplementation.scan_server(server_info)
        assert not result.supports_early_data

    @can_only_run_on_linux_64
    def test_early_data_disabled(self) -> None:
        if False:
            return 10
        with ModernOpenSslServer(max_early_data=None) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: EarlyDataScanResult = EarlyDataImplementation.scan_server(server_info)
        assert not result.supports_early_data