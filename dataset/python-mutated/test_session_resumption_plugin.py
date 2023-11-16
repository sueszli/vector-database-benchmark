import pytest
from nassl.ssl_client import ClientCertificateRequested
from sslyze import TlsResumptionSupportEnum
from sslyze.plugins.session_resumption.implementation import SessionResumptionSupportImplementation, SessionResumptionSupportScanResult, SessionResumptionSupportExtraArgument
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration, ClientAuthenticationCredentials
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ModernOpenSslServer, ClientAuthConfigEnum, LegacyOpenSslServer

class TestSessionResumptionSupport:

    def test(self) -> None:
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('www.google.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: SessionResumptionSupportScanResult = SessionResumptionSupportImplementation.scan_server(server_info)
        assert result.session_id_resumption_result == TlsResumptionSupportEnum.FULLY_SUPPORTED
        assert result.session_id_attempted_resumptions_count
        assert result.session_id_successful_resumptions_count
        assert result.tls_ticket_resumption_result == TlsResumptionSupportEnum.FULLY_SUPPORTED
        assert result.tls_ticket_attempted_resumptions_count
        assert result.tls_ticket_successful_resumptions_count
        assert SessionResumptionSupportImplementation.cli_connector_cls.result_to_console_output(result)

    def test_with_extra_argument(self) -> None:
        if False:
            while True:
                i = 10
        server_location = ServerNetworkLocation('www.google.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        custom_resumption_attempts_count = 6
        extra_arg = SessionResumptionSupportExtraArgument(number_of_resumptions_to_attempt=custom_resumption_attempts_count)
        result: SessionResumptionSupportScanResult = SessionResumptionSupportImplementation.scan_server(server_info, extra_arguments=extra_arg)
        assert result.session_id_attempted_resumptions_count == custom_resumption_attempts_count
        assert result.tls_ticket_attempted_resumptions_count == custom_resumption_attempts_count
        assert SessionResumptionSupportImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_fails_when_client_auth_failed(self):
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            with pytest.raises(ClientCertificateRequested):
                SessionResumptionSupportImplementation.scan_server(server_info)

    @can_only_run_on_linux_64
    def test_works_when_client_auth_succeeded(self) -> None:
        if False:
            i = 10
            return i + 15
        with ModernOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            network_config = ServerNetworkConfiguration(tls_server_name_indication=server.hostname, tls_client_auth_credentials=ClientAuthenticationCredentials(certificate_chain_path=server.get_client_certificate_path(), key_path=server.get_client_key_path()))
            server_info = check_connectivity_to_server_and_return_info(server_location, network_config)
            result: SessionResumptionSupportScanResult = SessionResumptionSupportImplementation.scan_server(server_info)
        assert result.session_id_successful_resumptions_count
        assert result.session_id_resumption_result == TlsResumptionSupportEnum.FULLY_SUPPORTED