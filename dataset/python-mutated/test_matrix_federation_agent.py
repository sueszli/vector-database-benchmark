import base64
import logging
import os
from typing import Generator, List, Optional, cast
from unittest.mock import AsyncMock, call, patch
import treq
from netaddr import IPSet
from service_identity import VerificationError
from zope.interface import implementer
from twisted.internet import defer
from twisted.internet._sslverify import ClientTLSOptions, OpenSSLCertificateOptions
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import _WrappingProtocol
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator, IProtocolFactory
from twisted.internet.protocol import Factory, Protocol
from twisted.protocols.tls import TLSMemoryBIOProtocol
from twisted.web._newclient import ResponseNeverReceived
from twisted.web.client import Agent
from twisted.web.http import HTTPChannel, Request
from twisted.web.http_headers import Headers
from twisted.web.iweb import IPolicyForHTTPS, IResponse
from synapse.config.homeserver import HomeServerConfig
from synapse.crypto.context_factory import FederationPolicyForHTTPS
from synapse.http.federation.matrix_federation_agent import MatrixFederationAgent
from synapse.http.federation.srv_resolver import Server, SrvResolver
from synapse.http.federation.well_known_resolver import WELL_KNOWN_MAX_SIZE, WellKnownResolver, _cache_period_from_headers
from synapse.logging.context import SENTINEL_CONTEXT, LoggingContext, LoggingContextOrSentinel, current_context
from synapse.types import ISynapseReactor
from synapse.util.caches.ttlcache import TTLCache
from tests import unittest
from tests.http import dummy_address, get_test_ca_cert_file, wrap_server_factory_for_tls
from tests.server import FakeTransport, ThreadedMemoryReactorClock
from tests.utils import checked_cast, default_config
logger = logging.getLogger(__name__)

class MatrixFederationAgentTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.reactor = ThreadedMemoryReactorClock()
        self.mock_resolver = AsyncMock(spec=SrvResolver)
        config_dict = default_config('test', parse=False)
        config_dict['federation_custom_ca_list'] = [get_test_ca_cert_file()]
        self._config = config = HomeServerConfig()
        config.parse_config_dict(config_dict, '', '')
        self.tls_factory = FederationPolicyForHTTPS(config)
        self.well_known_cache: TTLCache[bytes, Optional[bytes]] = TTLCache('test_cache', timer=self.reactor.seconds)
        self.had_well_known_cache: TTLCache[bytes, bool] = TTLCache('test_cache', timer=self.reactor.seconds)
        self.well_known_resolver = WellKnownResolver(self.reactor, Agent(self.reactor, contextFactory=self.tls_factory), b'test-agent', well_known_cache=self.well_known_cache, had_well_known_cache=self.had_well_known_cache)

    def _make_connection(self, client_factory: IProtocolFactory, ssl: bool=True, expected_sni: Optional[bytes]=None, tls_sanlist: Optional[List[bytes]]=None) -> HTTPChannel:
        if False:
            i = 10
            return i + 15
        'Builds a test server, and completes the outgoing client connection\n        Args:\n            client_factory: the the factory that the\n                application is trying to use to make the outbound connection. We will\n                invoke it to build the client Protocol\n\n            ssl: If true, we will expect an ssl connection and wrap\n                server_factory with a TLSMemoryBIOFactory\n                False is set only for when proxy expect http connection.\n                Otherwise federation requests use always https.\n\n            expected_sni: the expected SNI value\n\n            tls_sanlist: list of SAN entries for the TLS cert presented by the server.\n\n        Returns:\n            the server Protocol returned by server_factory\n        '
        server_factory = _get_test_protocol_factory()
        if ssl:
            server_factory = wrap_server_factory_for_tls(server_factory, self.reactor, tls_sanlist or [b'DNS:testserv', b'DNS:target-server', b'DNS:xn--bcher-kva.com', b'IP:1.2.3.4', b'IP:::1'])
        server_protocol = server_factory.buildProtocol(dummy_address)
        assert server_protocol is not None
        client_protocol = checked_cast(_WrappingProtocol, client_factory.buildProtocol(dummy_address))
        client_protocol.makeConnection(FakeTransport(server_protocol, self.reactor, client_protocol))
        server_protocol.makeConnection(FakeTransport(client_protocol, self.reactor, server_protocol))
        if ssl:
            assert isinstance(server_protocol, TLSMemoryBIOProtocol)
            http_protocol = server_protocol.wrappedProtocol
            tls_connection = server_protocol._tlsConnection
        else:
            http_protocol = server_protocol
            tls_connection = None
        assert isinstance(http_protocol, HTTPChannel)
        self.reactor.advance(0)
        if expected_sni is not None:
            server_name = tls_connection.get_servername()
            self.assertEqual(server_name, expected_sni, f'Expected SNI {expected_sni!s} but got {server_name!s}')
        return http_protocol

    @defer.inlineCallbacks
    def _make_get_request(self, uri: bytes) -> Generator['Deferred[object]', object, IResponse]:
        if False:
            while True:
                i = 10
        '\n        Sends a simple GET request via the agent, and checks its logcontext management\n        '
        with LoggingContext('one') as context:
            fetch_d: Deferred[IResponse] = self.agent.request(b'GET', uri)
            self.assertNoResult(fetch_d)
            _check_logcontext(SENTINEL_CONTEXT)
            fetch_res: IResponse
            try:
                fetch_res = (yield fetch_d)
                return fetch_res
            except Exception as e:
                logger.info('Fetch of %s failed: %s', uri.decode('ascii'), e)
                raise
            finally:
                _check_logcontext(context)

    def _handle_well_known_connection(self, client_factory: IProtocolFactory, expected_sni: bytes, content: bytes, response_headers: Optional[dict]=None) -> HTTPChannel:
        if False:
            i = 10
            return i + 15
        'Handle an outgoing HTTPs connection: wire it up to a server, check that the\n        request is for a .well-known, and send the response.\n\n        Args:\n            client_factory: outgoing connection\n            expected_sni: SNI that we expect the outgoing connection to send\n            content: content to send back as the .well-known\n        Returns:\n            server impl\n        '
        well_known_server = self._make_connection(client_factory, expected_sni=expected_sni)
        self.assertEqual(len(well_known_server.requests), 1)
        request = well_known_server.requests[0]
        self.assertEqual(request.requestHeaders.getRawHeaders(b'user-agent'), [b'test-agent'])
        self._send_well_known_response(request, content, headers=response_headers or {})
        return well_known_server

    def _send_well_known_response(self, request: Request, content: bytes, headers: Optional[dict]=None) -> None:
        if False:
            print('Hello World!')
        'Check that an incoming request looks like a valid .well-known request, and\n        send back the response.\n        '
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/.well-known/matrix/server')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        for (k, v) in (headers or {}).items():
            request.setHeader(k, v)
        request.write(content)
        request.finish()
        self.reactor.pump((0.1,))

    def _make_agent(self) -> MatrixFederationAgent:
        if False:
            i = 10
            return i + 15
        '\n        If a proxy server is set, the MatrixFederationAgent must be created again\n        because it is created too early during setUp\n        '
        return MatrixFederationAgent(reactor=cast(ISynapseReactor, self.reactor), tls_client_options_factory=self.tls_factory, user_agent=b'test-agent', ip_allowlist=IPSet(), ip_blocklist=IPSet(), _srv_resolver=self.mock_resolver, _well_known_resolver=self.well_known_resolver)

    def test_get(self) -> None:
        if False:
            while True:
                i = 10
        'happy-path test of a GET request with an explicit port'
        self._do_get()

    @patch.dict(os.environ, {'https_proxy': 'proxy.com', 'no_proxy': 'testserv'})
    def test_get_bypass_proxy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'test of a GET request with an explicit port and bypass proxy'
        self._do_get()

    def _do_get(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'test of a GET request with an explicit port'
        self.agent = self._make_agent()
        self.reactor.lookups['testserv'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv:8448/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv:8448'])
        self.assertEqual(request.requestHeaders.getRawHeaders(b'user-agent'), [b'test-agent'])
        content = request.content.read()
        self.assertEqual(content, b'')
        self.assertNoResult(test_d)
        request.responseHeaders.setRawHeaders(b'Content-Type', [b'application/json'])
        request.write('')
        self.reactor.pump((0.1,))
        response = self.successResultOf(test_d)
        self.assertEqual(response.code, 200)
        request.write(b'{ "a": 1 }')
        request.finish()
        self.reactor.pump((0.1,))
        json = self.successResultOf(treq.json_content(response))
        self.assertEqual(json, {'a': 1})

    @patch.dict(os.environ, {'https_proxy': 'http://proxy.com', 'no_proxy': 'unused.com'})
    def test_get_via_http_proxy(self) -> None:
        if False:
            return 10
        'test for federation request through a http proxy'
        self._do_get_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=None)

    @patch.dict(os.environ, {'https_proxy': 'http://user:pass@proxy.com', 'no_proxy': 'unused.com'})
    def test_get_via_http_proxy_with_auth(self) -> None:
        if False:
            i = 10
            return i + 15
        'test for federation request through a http proxy with authentication'
        self._do_get_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=b'user:pass')

    @patch.dict(os.environ, {'https_proxy': 'https://proxy.com', 'no_proxy': 'unused.com'})
    def test_get_via_https_proxy(self) -> None:
        if False:
            i = 10
            return i + 15
        'test for federation request through a https proxy'
        self._do_get_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=None)

    @patch.dict(os.environ, {'https_proxy': 'https://user:pass@proxy.com', 'no_proxy': 'unused.com'})
    def test_get_via_https_proxy_with_auth(self) -> None:
        if False:
            return 10
        'test for federation request through a https proxy with authentication'
        self._do_get_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=b'user:pass')

    def _do_get_via_proxy(self, expect_proxy_ssl: bool=False, expected_auth_credentials: Optional[bytes]=None) -> None:
        if False:
            while True:
                i = 10
        'Send a https federation request via an agent and check that it is correctly\n            received at the proxy and client. The proxy can use either http or https.\n        Args:\n            expect_proxy_ssl: True if we expect the request to connect to the proxy via https.\n            expected_auth_credentials: credentials we expect to be presented to authenticate at the proxy\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.reactor.lookups['proxy.com'] = '9.9.9.9'
        test_d = self._make_get_request(b'matrix-federation://testserv:8448/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '9.9.9.9')
        self.assertEqual(port, 1080)
        proxy_server = self._make_connection(client_factory, ssl=expect_proxy_ssl, tls_sanlist=[b'DNS:proxy.com'] if expect_proxy_ssl else None, expected_sni=b'proxy.com' if expect_proxy_ssl else None)
        assert isinstance(proxy_server, HTTPChannel)
        self.assertEqual(len(proxy_server.requests), 1)
        request = proxy_server.requests[0]
        self.assertEqual(request.method, b'CONNECT')
        self.assertEqual(request.path, b'testserv:8448')
        proxy_auth_header_values = request.requestHeaders.getRawHeaders(b'Proxy-Authorization')
        if expected_auth_credentials is not None:
            encoded_credentials = base64.b64encode(expected_auth_credentials)
            expected_header_value = b'Basic ' + encoded_credentials
            self.assertIn(expected_header_value, proxy_auth_header_values)
        else:
            self.assertIsNone(proxy_auth_header_values)
        proxy_server.persistent = True
        request.finish()
        server_ssl_protocol = wrap_server_factory_for_tls(_get_test_protocol_factory(), self.reactor, sanlist=[b'DNS:testserv', b'DNS:target-server', b'DNS:xn--bcher-kva.com', b'IP:1.2.3.4', b'IP:::1']).buildProtocol(dummy_address)
        proxy_server_transport = proxy_server.transport
        assert proxy_server_transport is not None
        server_ssl_protocol.makeConnection(proxy_server_transport)
        if expect_proxy_ssl:
            assert isinstance(proxy_server_transport, TLSMemoryBIOProtocol)
            proxy_server_transport.wrappedProtocol = server_ssl_protocol
        else:
            assert isinstance(proxy_server_transport, FakeTransport)
            client_protocol = proxy_server_transport.other
            assert isinstance(client_protocol, Protocol)
            c2s_transport = checked_cast(FakeTransport, client_protocol.transport)
            c2s_transport.other = server_ssl_protocol
        self.reactor.advance(0)
        server_name = server_ssl_protocol._tlsConnection.get_servername()
        expected_sni = b'testserv'
        self.assertEqual(server_name, expected_sni, f'Expected SNI {expected_sni!s} but got {server_name!s}')
        http_server = server_ssl_protocol.wrappedProtocol
        assert isinstance(http_server, HTTPChannel)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv:8448'])
        self.assertEqual(request.requestHeaders.getRawHeaders(b'user-agent'), [b'test-agent'])
        self.assertIsNone(request.requestHeaders.getRawHeaders(b'Proxy-Authorization'))
        content = request.content.read()
        self.assertEqual(content, b'')
        self.assertNoResult(test_d)
        request.responseHeaders.setRawHeaders(b'Content-Type', [b'application/json'])
        request.write('')
        self.reactor.pump((0.1,))
        response = self.successResultOf(test_d)
        self.assertEqual(response.code, 200)
        request.write(b'{ "a": 1 }')
        request.finish()
        self.reactor.pump((0.1,))
        json = self.successResultOf(treq.json_content(response))
        self.assertEqual(json, {'a': 1})

    def test_get_ip_address(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test the behaviour when the server name contains an explicit IP (with no port)\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['1.2.3.4'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://1.2.3.4/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=None)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'1.2.3.4'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_ipv6_address(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the behaviour when the server name contains an explicit IPv6 address\n        (with no port)\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['::1'] = '::1'
        test_d = self._make_get_request(b'matrix-federation://[::1]/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '::1')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=None)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'[::1]'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_ipv6_address_with_port(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test the behaviour when the server name contains an explicit IPv6 address\n        (with explicit port)\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['::1'] = '::1'
        test_d = self._make_get_request(b'matrix-federation://[::1]:80/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '::1')
        self.assertEqual(port, 80)
        http_server = self._make_connection(client_factory, expected_sni=None)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'[::1]:80'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_hostname_bad_cert(self) -> None:
        if False:
            print('Hello World!')
        "\n        Test the behaviour when the certificate on the server doesn't match the hostname\n        "
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv1'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv1/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_not_called()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv1'), call(b'_matrix._tcp.testserv1')])
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv1')
        self.assertEqual(len(http_server.requests), 0)
        e = self.failureResultOf(test_d, ResponseNeverReceived)
        failure_reason = e.value.reasons[0]
        self.assertIsInstance(failure_reason.value, VerificationError)

    def test_get_ip_address_bad_cert(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the behaviour when the server name contains an explicit IP, but\n        the server cert doesn't cover it\n        "
        self.agent = self._make_agent()
        self.reactor.lookups['1.2.3.5'] = '1.2.3.5'
        test_d = self._make_get_request(b'matrix-federation://1.2.3.5/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.5')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=None)
        self.assertEqual(len(http_server.requests), 0)
        e = self.failureResultOf(test_d, ResponseNeverReceived)
        failure_reason = e.value.reasons[0]
        self.assertIsInstance(failure_reason.value, VerificationError)

    def test_get_no_srv_no_well_known(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the behaviour when the server name has no port, no SRV, and no well-known\n        '
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_not_called()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv'), call(b'_matrix._tcp.testserv')])
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_well_known(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the behaviour when the .well-known delegates elsewhere'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.reactor.lookups['target-server'] = '1::f'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', content=b'{ "m.server": "target-server" }')
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.target-server'), call(b'_matrix._tcp.target-server')])
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '1::f')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'target-server')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'target-server'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)
        self.assertEqual(self.well_known_cache[b'testserv'], b'target-server')
        self.reactor.pump((48 * 3600,))
        self.well_known_cache.expire()
        self.assertNotIn(b'testserv', self.well_known_cache)

    def test_get_well_known_redirect(self) -> None:
        if False:
            print('Hello World!')
        'Test the behaviour when the server name has no port and no SRV record, but\n        the .well-known has a 300 redirect\n        '
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.reactor.lookups['target-server'] = '1::f'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop()
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        redirect_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(redirect_server.requests), 1)
        request = redirect_server.requests[0]
        request.redirect(b'https://testserv/even_better_known')
        request.finish()
        self.reactor.pump((0.1,))
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop()
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        well_known_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(well_known_server.requests), 1, 'No request after 302')
        request = well_known_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/even_better_known')
        request.write(b'{ "m.server": "target-server" }')
        request.finish()
        self.reactor.pump((0.1,))
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.target-server'), call(b'_matrix._tcp.target-server')])
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1::f')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'target-server')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'target-server'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)
        self.assertEqual(self.well_known_cache[b'testserv'], b'target-server')
        self.reactor.pump((48 * 3600,))
        self.well_known_cache.expire()
        self.assertNotIn(b'testserv', self.well_known_cache)

    def test_get_invalid_well_known(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test the behaviour when the server name has an *invalid* well-known (and no SRV)\n        '
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_not_called()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop()
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', content=b'NOT JSON')
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv'), call(b'_matrix._tcp.testserv')])
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop()
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_well_known_unsigned_cert(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the behaviour when the .well-known server presents a cert\n        not signed by a CA\n        '
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['testserv'] = '1.2.3.4'
        config = default_config('test', parse=True)
        tls_factory = FederationPolicyForHTTPS(config)
        agent = MatrixFederationAgent(reactor=self.reactor, tls_client_options_factory=tls_factory, user_agent=b'test-agent', ip_allowlist=IPSet(), ip_blocklist=IPSet(), _srv_resolver=self.mock_resolver, _well_known_resolver=WellKnownResolver(cast(ISynapseReactor, self.reactor), Agent(self.reactor, contextFactory=tls_factory), b'test-agent', well_known_cache=self.well_known_cache, had_well_known_cache=self.had_well_known_cache))
        test_d = agent.request(b'GET', b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        http_proto = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_proto.requests), 0)
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv'), call(b'_matrix._tcp.testserv')])

    def test_get_hostname_srv(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test the behaviour when there is a single SRV record for _matrix-fed.\n        '
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = [Server(host=b'srvtarget', port=8443)]
        self.reactor.lookups['srvtarget'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_called_once_with(b'_matrix-fed._tcp.testserv')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_hostname_srv_legacy(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the behaviour when there is a single SRV record for _matrix.\n        '
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.side_effect = [[], [Server(host=b'srvtarget', port=8443)]]
        self.reactor.lookups['srvtarget'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv'), call(b'_matrix._tcp.testserv')])
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_well_known_srv(self) -> None:
        if False:
            return 10
        'Test the behaviour when the .well-known redirects to a place where there\n        is a _matrix-fed SRV record.\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.reactor.lookups['srvtarget'] = '5.6.7.8'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self.mock_resolver.resolve_service.return_value = [Server(host=b'srvtarget', port=8443)]
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', content=b'{ "m.server": "target-server" }')
        self.mock_resolver.resolve_service.assert_called_once_with(b'_matrix-fed._tcp.target-server')
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '5.6.7.8')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'target-server')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'target-server'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_get_well_known_srv_legacy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the behaviour when the .well-known redirects to a place where there\n        is a _matrix SRV record.\n        '
        self.agent = self._make_agent()
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.reactor.lookups['srvtarget'] = '5.6.7.8'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self.mock_resolver.resolve_service.side_effect = [[], [Server(host=b'srvtarget', port=8443)]]
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', content=b'{ "m.server": "target-server" }')
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.target-server'), call(b'_matrix._tcp.target-server')])
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '5.6.7.8')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'target-server')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'target-server'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_idna_servername(self) -> None:
        if False:
            print('Hello World!')
        'test the behaviour when the server name has idna chars in'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = []
        self.reactor.lookups['xn--bcher-kva.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://xn--bcher-kva.com/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_not_called()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.xn--bcher-kva.com'), call(b'_matrix._tcp.xn--bcher-kva.com')])
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 2)
        (host, port, client_factory, _timeout, _bindAddress) = clients[1]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8448)
        http_server = self._make_connection(client_factory, expected_sni=b'xn--bcher-kva.com')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'xn--bcher-kva.com'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_idna_srv_target(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'test the behaviour when the target of a _matrix-fed SRV record has idna chars'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = [Server(host=b'xn--trget-3qa.com', port=8443)]
        self.reactor.lookups['xn--trget-3qa.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://xn--bcher-kva.com/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_called_once_with(b'_matrix-fed._tcp.xn--bcher-kva.com')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'xn--bcher-kva.com')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'xn--bcher-kva.com'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_idna_srv_target_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        'test the behaviour when the target of a _matrix SRV record has idna chars'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.side_effect = [[], [Server(host=b'xn--trget-3qa.com', port=8443)]]
        self.reactor.lookups['xn--trget-3qa.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://xn--bcher-kva.com/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.xn--bcher-kva.com'), call(b'_matrix._tcp.xn--bcher-kva.com')])
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        http_server = self._make_connection(client_factory, expected_sni=b'xn--bcher-kva.com')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'xn--bcher-kva.com'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_well_known_cache(self) -> None:
        if False:
            return 10
        self.reactor.lookups['testserv'] = '1.2.3.4'
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        well_known_server = self._handle_well_known_connection(client_factory, expected_sni=b'testserv', response_headers={b'Cache-Control': b'max-age=1000'}, content=b'{ "m.server": "target-server" }')
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, b'target-server')
        well_known_server.loseConnection()
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, b'target-server')
        self.reactor.pump((1000.0,))
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', content=b'{ "m.server": "other-server" }')
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, b'other-server')

    def test_well_known_cache_with_temp_failure(self) -> None:
        if False:
            print('Hello World!')
        'Test that we refetch well-known before the cache expires, and that\n        it ignores transient errors.\n        '
        self.reactor.lookups['testserv'] = '1.2.3.4'
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        well_known_server = self._handle_well_known_connection(client_factory, expected_sni=b'testserv', response_headers={b'Cache-Control': b'max-age=1000'}, content=b'{ "m.server": "target-server" }')
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, b'target-server')
        well_known_server.loseConnection()
        self.reactor.pump((900.0,))
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        attempts = 0
        while self.reactor.tcpClients:
            clients = self.reactor.tcpClients
            (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
            attempts += 1
            client_factory.clientConnectionFailed(None, Exception('nope'))
            self.reactor.pump((1.0, 1.0))
        self.assertGreater(attempts, 1)
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, b'target-server')
        self.reactor.pump((10000.0,))
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        clients = self.reactor.tcpClients
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        r = self.successResultOf(fetch_d)
        self.assertEqual(r.delegated_server, None)

    def test_well_known_too_large(self) -> None:
        if False:
            i = 10
            return i + 15
        'A well-known query that returns a result which is too large should be rejected.'
        self.reactor.lookups['testserv'] = '1.2.3.4'
        fetch_d = defer.ensureDeferred(self.well_known_resolver.get_well_known(b'testserv'))
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443)
        self._handle_well_known_connection(client_factory, expected_sni=b'testserv', response_headers={b'Cache-Control': b'max-age=1000'}, content=b'{ "m.server": "' + b'a' * WELL_KNOWN_MAX_SIZE + b'" }')
        r = self.successResultOf(fetch_d)
        self.assertIsNone(r.delegated_server)

    def test_srv_fallbacks(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that other SRV results are tried if the first one fails for _matrix-fed SRV.'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.return_value = [Server(host=b'target.com', port=8443), Server(host=b'target.com', port=8444)]
        self.reactor.lookups['target.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_called_once_with(b'_matrix-fed._tcp.testserv')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8444)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_srv_fallbacks_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that other SRV results are tried if the first one fails for _matrix SRV.'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.side_effect = [[], [Server(host=b'target.com', port=8443), Server(host=b'target.com', port=8444)]]
        self.reactor.lookups['target.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_has_calls([call(b'_matrix-fed._tcp.testserv'), call(b'_matrix._tcp.testserv')])
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8444)
        http_server = self._make_connection(client_factory, expected_sni=b'testserv')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/foo/bar')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'testserv'])
        request.finish()
        self.reactor.pump((0.1,))
        self.successResultOf(test_d)

    def test_srv_no_fallback_to_legacy(self) -> None:
        if False:
            return 10
        'Test that _matrix SRV results are not tried if the _matrix-fed one fails.'
        self.agent = self._make_agent()
        self.mock_resolver.resolve_service.side_effect = [[Server(host=b'target.com', port=8443)], []]
        self.reactor.lookups['target.com'] = '1.2.3.4'
        test_d = self._make_get_request(b'matrix-federation://testserv/foo/bar')
        self.assertNoResult(test_d)
        self.mock_resolver.resolve_service.assert_called_once_with(b'_matrix-fed._tcp.testserv')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop(0)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8443)
        client_factory.clientConnectionFailed(None, Exception('nope'))
        self.reactor.pump((0.4,))
        self.assertFailure(test_d, Exception)

class TestCachePeriodFromHeaders(unittest.TestCase):

    def test_cache_control(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(_cache_period_from_headers(Headers({b'Cache-Control': [b'foo, Max-Age = 100, bar']})), 100)
        self.assertIsNone(_cache_period_from_headers(Headers({b'Cache-Control': [b'max-age=, bar']})))
        self.assertIsNone(_cache_period_from_headers(Headers({b'Cache-Control': [b'private; max-age=0']})))
        self.assertEqual(_cache_period_from_headers(Headers({b'Cache-Control': [b'max-age=0, private, must-revalidate']})), 0)
        self.assertEqual(_cache_period_from_headers(Headers({b'cache-control': [b'private, max-age=0']})), 0)

    def test_expires(self) -> None:
        if False:
            return 10
        self.assertEqual(_cache_period_from_headers(Headers({b'Expires': [b'Wed, 30 Jan 2019 07:35:33 GMT']}), time_now=lambda : 1548833700), 33)
        self.assertEqual(_cache_period_from_headers(Headers({b'cache-control': [b'max-age=10'], b'Expires': [b'Wed, 30 Jan 2019 07:35:33 GMT']}), time_now=lambda : 1548833700), 10)
        self.assertEqual(_cache_period_from_headers(Headers({b'Expires': [b'0']})), 0)

def _check_logcontext(context: LoggingContextOrSentinel) -> None:
    if False:
        i = 10
        return i + 15
    current = current_context()
    if current is not context:
        raise AssertionError('Expected logcontext %s but was %s' % (context, current))

def _get_test_protocol_factory() -> IProtocolFactory:
    if False:
        i = 10
        return i + 15
    'Get a protocol Factory which will build an HTTPChannel\n    Returns:\n        interfaces.IProtocolFactory\n    '
    server_factory = Factory.forProtocol(HTTPChannel)
    server_factory.log = _log_request
    return server_factory

def _log_request(request: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Implements Factory.log, which is expected by Request.finish'
    logger.info(f'Completed request {request}')

@implementer(IPolicyForHTTPS)
class TrustingTLSPolicyForHTTPS:
    """An IPolicyForHTTPS which checks that the certificate belongs to the
    right server, but doesn't check the certificate chain."""

    def creatorForNetloc(self, hostname: bytes, port: int) -> IOpenSSLClientConnectionCreator:
        if False:
            for i in range(10):
                print('nop')
        certificateOptions = OpenSSLCertificateOptions()
        return ClientTLSOptions(hostname, certificateOptions.getContext())