from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Dict
import pytest
from nassl.ssl_client import ClientCertificateRequested
from sslyze.plugins.http_headers_plugin import HttpHeadersImplementation, HttpHeadersScanResult, _detect_http_redirection, HttpHeadersScanResultAsJson
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration, ClientAuthenticationCredentials
from tests.connectivity_utils import check_connectivity_to_server_and_return_info
from tests.markers import can_only_run_on_linux_64
from tests.openssl_server import ClientAuthConfigEnum, LegacyOpenSslServer, ModernOpenSslServer

class TestHttpHeadersPlugin:

    def test_hsts_enabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation('hsts.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: HttpHeadersScanResult = HttpHeadersImplementation.scan_server(server_info)
        assert result.http_request_sent
        assert result.http_path_redirected_to
        assert result.strict_transport_security_header
        assert HttpHeadersImplementation.cli_connector_cls.result_to_console_output(result)

    def test_all_headers_disabled(self) -> None:
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation('expired.badssl.com', 443)
        server_info = check_connectivity_to_server_and_return_info(server_location)
        result: HttpHeadersScanResult = HttpHeadersImplementation.scan_server(server_info)
        assert result.http_request_sent
        assert result.http_path_redirected_to
        assert not result.strict_transport_security_header
        assert not result.expect_ct_header
        assert HttpHeadersImplementation.cli_connector_cls.result_to_console_output(result)

    @can_only_run_on_linux_64
    def test_http_error(self) -> None:
        if False:
            i = 10
            return i + 15
        with ModernOpenSslServer(should_reply_to_http_requests=False) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            result: HttpHeadersScanResult = HttpHeadersImplementation.scan_server(server_info)
        assert result.http_error_trace
        assert result.http_request_sent
        assert not result.http_path_redirected_to
        assert HttpHeadersImplementation.cli_connector_cls.result_to_console_output(result)
        result_as_json = HttpHeadersScanResultAsJson.from_orm(result).json()
        assert result_as_json

    @can_only_run_on_linux_64
    def test_fails_when_client_auth_failed(self) -> None:
        if False:
            return 10
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            server_info = check_connectivity_to_server_and_return_info(server_location)
            with pytest.raises(ClientCertificateRequested):
                HttpHeadersImplementation.scan_server(server_info)

    @can_only_run_on_linux_64
    def test_works_when_client_auth_succeeded(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LegacyOpenSslServer(client_auth_config=ClientAuthConfigEnum.REQUIRED) as server:
            server_location = ServerNetworkLocation(hostname=server.hostname, ip_address=server.ip_address, port=server.port)
            network_config = ServerNetworkConfiguration(tls_server_name_indication=server.hostname, tls_client_auth_credentials=ClientAuthenticationCredentials(certificate_chain_path=server.get_client_certificate_path(), key_path=server.get_client_key_path()))
            server_info = check_connectivity_to_server_and_return_info(server_location, network_config)
            result: HttpHeadersScanResult = HttpHeadersImplementation.scan_server(server_info)
            assert not result.strict_transport_security_header

@dataclass
class _MockHttpResponse(HTTPResponse):
    status: int
    _headers: Dict[str, str]

    def getheader(self, name: str, default=None):
        if False:
            print('Hello World!')
        "Replicate HTTPResponse's API."
        return self._headers[name]

class TestHttpRedirection:

    def test_no_redirection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        http_response = _MockHttpResponse(status=200, _headers={})
        next_location_path = _detect_http_redirection(http_response=http_response, server_host_name='lol.com', server_port=443)
        assert next_location_path is None

    def test_redirection_relative_url(self) -> None:
        if False:
            while True:
                i = 10
        http_response = _MockHttpResponse(status=302, _headers={'Location': '/newpath'})
        next_location_path = _detect_http_redirection(http_response=http_response, server_host_name='lol.com', server_port=443)
        assert next_location_path == '/newpath'

    def test_redirection_absolute_url_same_server(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        http_response = _MockHttpResponse(status=302, _headers={'Location': 'https://lol.com/newpath'})
        next_location_path = _detect_http_redirection(http_response=http_response, server_host_name='lol.com', server_port=443)
        assert next_location_path == '/newpath'

    def test_redirection_absolute_url_different_hostname(self) -> None:
        if False:
            print('Hello World!')
        http_response = _MockHttpResponse(status=302, _headers={'Location': 'https://otherdomain.com/newpath'})
        next_location_path = _detect_http_redirection(http_response=http_response, server_host_name='lol.com', server_port=443)
        assert next_location_path is None

    def test_redirection_absolute_url_different_port(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        http_response = _MockHttpResponse(status=302, _headers={'Location': 'https://lol.com:444/newpath'})
        next_location_path = _detect_http_redirection(http_response=http_response, server_host_name='lol.com', server_port=443)
        assert next_location_path is None