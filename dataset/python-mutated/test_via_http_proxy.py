import threading
import pytest
from sslyze.json.json_output import _ServerTlsProbingResultAsJson
from sslyze.server_connectivity import check_connectivity_to_server
from sslyze.server_setting import ServerNetworkLocation, HttpProxySettings, ServerNetworkConfiguration
from sslyze.errors import ConnectionToHttpProxyTimedOut, ConnectionToHttpProxyFailed, HttpProxyRejectedConnection
from tests.server_connectivity_tests.tiny_proxy import ThreadingHTTPServer, ProxyHandler

class TestServerConnectivityTesterWithProxy:

    def test_via_http_proxy(self):
        if False:
            return 10
        proxy_port = 8123
        proxy_server = ThreadingHTTPServer(('', proxy_port), ProxyHandler)
        proxy_server_thread = threading.Thread(target=proxy_server.serve_forever)
        proxy_server_thread.start()
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, http_proxy_settings=HttpProxySettings('localhost', proxy_port))
        try:
            tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))
        finally:
            proxy_server.shutdown()
        assert tls_probing_result.cipher_suite_supported
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.client_auth_requirement
        tls_probing_result_as_json = _ServerTlsProbingResultAsJson.from_orm(tls_probing_result)
        assert tls_probing_result_as_json.json()

    def test_via_http_proxy_but_proxy_dns_error(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, http_proxy_settings=HttpProxySettings('notarealdomain.not.real.notreal.not', 443))
        with pytest.raises(ConnectionToHttpProxyFailed):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_via_http_proxy_but_proxy_timed_out(self):
        if False:
            for i in range(10):
                print('nop')
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, http_proxy_settings=HttpProxySettings('1.2.3.4', 80))
        with pytest.raises(ConnectionToHttpProxyTimedOut):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_via_http_proxy_but_proxy_rejected_connection(self):
        if False:
            i = 10
            return i + 15
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, http_proxy_settings=HttpProxySettings('localhost', 1234))
        with pytest.raises(HttpProxyRejectedConnection):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))

    def test_via_http_proxy_but_proxy_rejected_http_connect(self):
        if False:
            return 10
        server_location = ServerNetworkLocation(hostname='www.google.com', port=443, http_proxy_settings=HttpProxySettings('www.hotmail.com', 443))
        with pytest.raises(HttpProxyRejectedConnection):
            check_connectivity_to_server(server_location=server_location, network_configuration=ServerNetworkConfiguration.default_for_server_location(server_location))