from nassl.ssl_client import ClientCertificateRequested
from sslyze.plugins.fallback_scsv_plugin import FallbackScsvScanResult, FallbackScsvImplementation
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration, ClientAuthenticationCredentials
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import LegacyOpenSslServer, ClientAuthConfigEnum
import pytest

class TestFallbackScsvPlugin:

    def test_fallback_good(self) -> None:
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.google.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: FallbackScsvScanResult = FallbackScsvImplementation.scan_server(server_info)
        assert result.supports_fallback_scsv
        assert FallbackScsvImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_fallback_bad(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer() as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: FallbackScsvScanResult = FallbackScsvImplementation.scan_server(server_info)
        assert not result.supports_fallback_scsv

    @can_only_run_on_linux_64
    def test_fails_when_client_auth_failed(self) -> None:
        if False:
            while True:
                i = 10
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            with pytest.raises(ClientCertificateRequested):
                FallbackScsvImplementation.scan_server(server_info)

    @can_only_run_on_linux_64
    def test_works_when_client_auth_succeeded(self) -> None:
        if False:
            while True:
                i = 10
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            network_config = ServerNetworkConfiguration(tls_server_name_indication=server.hostname, tls_client_auth_credentials=ClientAuthenticationCredentials(certificate_chain_path=server.get_client_certificate_path(), key_path=server.get_client_key_path()))
            server_info = check_connectivity_to_server_and_return_info(server_location, network_config)
            result: FallbackScsvScanResult = FallbackScsvImplementation.scan_server(server_info)
        assert not result.supports_fallback_scsv