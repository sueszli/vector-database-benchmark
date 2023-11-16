import base64
import logging
import os
from typing import List, Optional
from unittest.mock import patch
import treq
from netaddr import IPSet
from parameterized import parameterized
from twisted.internet import interfaces
from twisted.internet.endpoints import HostnameEndpoint, _WrapperEndpoint, _WrappingProtocol
from twisted.internet.interfaces import IProtocol, IProtocolFactory
from twisted.internet.protocol import Factory, Protocol
from twisted.protocols.tls import TLSMemoryBIOProtocol
from twisted.web.http import HTTPChannel
from synapse.http.client import BlocklistingReactorWrapper
from synapse.http.connectproxyclient import BasicProxyCredentials
from synapse.http.proxyagent import ProxyAgent, parse_proxy
from tests.http import dummy_address, get_test_https_policy, wrap_server_factory_for_tls
from tests.server import FakeTransport, ThreadedMemoryReactorClock
from tests.unittest import TestCase
from tests.utils import checked_cast
logger = logging.getLogger(__name__)
HTTPFactory = Factory.forProtocol(HTTPChannel)

class ProxyParserTests(TestCase):
    """
    Values for test
    [
        proxy_string,
        expected_scheme,
        expected_hostname,
        expected_port,
        expected_credentials,
    ]
    """

    @parameterized.expand([[b'localhost', b'http', b'localhost', 1080, None], [b'localhost:9988', b'http', b'localhost', 9988, None], [b'https://localhost', b'https', b'localhost', 1080, None], [b'https://localhost:1234', b'https', b'localhost', 1234, None], [b'1.2.3.4', b'http', b'1.2.3.4', 1080, None], [b'1.2.3.4:9988', b'http', b'1.2.3.4', 9988, None], [b'https://1.2.3.4', b'https', b'1.2.3.4', 1080, None], [b'https://1.2.3.4:9988', b'https', b'1.2.3.4', 9988, None], [b'[2001:0db8:85a3:0000:0000:8a2e:0370:effe]', b'http', b'2001:0db8:85a3:0000:0000:8a2e:0370:effe', 1080, None], [b'[2001:0db8:85a3:0000:0000:8a2e:0370:1234]', b'http', b'2001:0db8:85a3:0000:0000:8a2e:0370:1234', 1080, None], [b'[::1]', b'http', b'::1', 1080, None], [b'[::ffff:0.0.0.0]', b'http', b'::ffff:0.0.0.0', 1080, None], [b'[2001:0db8:85a3:0000:0000:8a2e:0370:effe]:9988', b'http', b'2001:0db8:85a3:0000:0000:8a2e:0370:effe', 9988, None], [b'[2001:0db8:85a3:0000:0000:8a2e:0370:1234]:9988', b'http', b'2001:0db8:85a3:0000:0000:8a2e:0370:1234', 9988, None], [b'[::1]:9988', b'http', b'::1', 9988, None], [b'[::ffff:0.0.0.0]:9988', b'http', b'::ffff:0.0.0.0', 9988, None], [b'https://[2001:0db8:85a3:0000:0000:8a2e:0370:effe]', b'https', b'2001:0db8:85a3:0000:0000:8a2e:0370:effe', 1080, None], [b'https://[2001:0db8:85a3:0000:0000:8a2e:0370:1234]', b'https', b'2001:0db8:85a3:0000:0000:8a2e:0370:1234', 1080, None], [b'https://[::1]', b'https', b'::1', 1080, None], [b'https://[::ffff:0.0.0.0]', b'https', b'::ffff:0.0.0.0', 1080, None], [b'https://[2001:0db8:85a3:0000:0000:8a2e:0370:effe]:9988', b'https', b'2001:0db8:85a3:0000:0000:8a2e:0370:effe', 9988, None], [b'https://[2001:0db8:85a3:0000:0000:8a2e:0370:1234]:9988', b'https', b'2001:0db8:85a3:0000:0000:8a2e:0370:1234', 9988, None], [b'https://[::1]:9988', b'https', b'::1', 9988, None], [b'https://user:pass@1.2.3.4:9988', b'https', b'1.2.3.4', 9988, b'user:pass'], [b'user:pass@1.2.3.4:9988', b'http', b'1.2.3.4', 9988, b'user:pass'], [b'https://user:pass@proxy.local:9988', b'https', b'proxy.local', 9988, b'user:pass'], [b'user:pass@proxy.local:9988', b'http', b'proxy.local', 9988, b'user:pass']])
    def test_parse_proxy(self, proxy_string: bytes, expected_scheme: bytes, expected_hostname: bytes, expected_port: int, expected_credentials: Optional[bytes]) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Tests that a given proxy URL will be broken into the components.\n        Args:\n            proxy_string: The proxy connection string.\n            expected_scheme: Expected value of proxy scheme.\n            expected_hostname: Expected value of proxy hostname.\n            expected_port: Expected value of proxy port.\n            expected_credentials: Expected value of credentials.\n                Must be in form '<username>:<password>' or None\n        "
        proxy_cred = None
        if expected_credentials:
            proxy_cred = BasicProxyCredentials(expected_credentials)
        self.assertEqual((expected_scheme, expected_hostname, expected_port, proxy_cred), parse_proxy(proxy_string))

class TestBasicProxyCredentials(TestCase):

    def test_long_user_pass_string_encoded_without_newlines(self) -> None:
        if False:
            i = 10
            return i + 15
        'Reproduces https://github.com/matrix-org/synapse/pull/16504.'
        proxy_connection_string = b'looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooonguser:pass@proxy.local:9988'
        (_, _, _, creds) = parse_proxy(proxy_connection_string)
        assert creds is not None
        self.assertIsInstance(creds, BasicProxyCredentials)
        auth_value = creds.as_proxy_authorization_value()
        self.assertNotIn(b'\n', auth_value)
        self.assertEqual(creds.as_proxy_authorization_value(), b'Basic bG9vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb29vbmd1c2VyOnBhc3M=')
        basic_auth_payload = creds.as_proxy_authorization_value().split(b' ')[1]
        self.assertEqual(base64.b64decode(basic_auth_payload), b'looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooonguser:pass')

class MatrixFederationAgentTests(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.reactor = ThreadedMemoryReactorClock()

    def _make_connection(self, client_factory: IProtocolFactory, server_factory: IProtocolFactory, ssl: bool=False, expected_sni: Optional[bytes]=None, tls_sanlist: Optional[List[bytes]]=None) -> IProtocol:
        if False:
            print('Hello World!')
        "Builds a test server, and completes the outgoing client connection\n\n        Args:\n            client_factory: the the factory that the\n                application is trying to use to make the outbound connection. We will\n                invoke it to build the client Protocol\n\n            server_factory: a factory to build the\n                server-side protocol\n\n            ssl: If true, we will expect an ssl connection and wrap\n                server_factory with a TLSMemoryBIOFactory\n\n            expected_sni: the expected SNI value\n\n            tls_sanlist: list of SAN entries for the TLS cert presented by the server.\n                 Defaults to [b'DNS:test.com']\n\n        Returns:\n            the server Protocol returned by server_factory\n        "
        if ssl:
            server_factory = wrap_server_factory_for_tls(server_factory, self.reactor, tls_sanlist or [b'DNS:test.com'])
        server_protocol = server_factory.buildProtocol(dummy_address)
        assert server_protocol is not None
        client_protocol = client_factory.buildProtocol(dummy_address)
        assert client_protocol is not None
        client_protocol.makeConnection(FakeTransport(server_protocol, self.reactor, client_protocol))
        server_protocol.makeConnection(FakeTransport(client_protocol, self.reactor, server_protocol))
        if ssl:
            assert isinstance(server_protocol, TLSMemoryBIOProtocol)
            http_protocol = server_protocol.wrappedProtocol
            tls_connection = server_protocol._tlsConnection
        else:
            http_protocol = server_protocol
            tls_connection = None
        self.reactor.advance(0)
        if expected_sni is not None:
            server_name = tls_connection.get_servername()
            self.assertEqual(server_name, expected_sni, f'Expected SNI {expected_sni!s} but got {server_name!s}')
        return http_protocol

    def _test_request_direct_connection(self, agent: ProxyAgent, scheme: bytes, hostname: bytes, path: bytes) -> None:
        if False:
            print('Hello World!')
        'Runs a test case for a direct connection not going through a proxy.\n\n        Args:\n            agent: the proxy agent being tested\n\n            scheme: expected to be either "http" or "https"\n\n            hostname: the hostname to connect to in the test\n\n            path: the path to connect to in the test\n        '
        is_https = scheme == b'https'
        self.reactor.lookups[hostname.decode()] = '1.2.3.4'
        d = agent.request(b'GET', scheme + b'://' + hostname + b'/' + path)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 443 if is_https else 80)
        http_server = self._make_connection(client_factory, _get_test_protocol_factory(), ssl=is_https, expected_sni=hostname if is_https else None)
        assert isinstance(http_server, HTTPChannel)
        self.reactor.advance(0)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/' + path)
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [hostname])
        request.write(b'result')
        request.finish()
        self.reactor.advance(0)
        resp = self.successResultOf(d)
        body = self.successResultOf(treq.content(resp))
        self.assertEqual(body, b'result')

    def test_http_request(self) -> None:
        if False:
            i = 10
            return i + 15
        agent = ProxyAgent(self.reactor)
        self._test_request_direct_connection(agent, b'http', b'test.com', b'')

    def test_https_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        agent = ProxyAgent(self.reactor, contextFactory=get_test_https_policy())
        self._test_request_direct_connection(agent, b'https', b'test.com', b'abc')

    def test_http_request_use_proxy_empty_environment(self) -> None:
        if False:
            print('Hello World!')
        agent = ProxyAgent(self.reactor, use_proxy=True)
        self._test_request_direct_connection(agent, b'http', b'test.com', b'')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888', 'NO_PROXY': 'test.com'})
    def test_http_request_via_uppercase_no_proxy(self) -> None:
        if False:
            print('Hello World!')
        agent = ProxyAgent(self.reactor, use_proxy=True)
        self._test_request_direct_connection(agent, b'http', b'test.com', b'')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888', 'no_proxy': 'test.com,unused.com'})
    def test_http_request_via_no_proxy(self) -> None:
        if False:
            while True:
                i = 10
        agent = ProxyAgent(self.reactor, use_proxy=True)
        self._test_request_direct_connection(agent, b'http', b'test.com', b'')

    @patch.dict(os.environ, {'https_proxy': 'proxy.com', 'no_proxy': 'test.com,unused.com'})
    def test_https_request_via_no_proxy(self) -> None:
        if False:
            print('Hello World!')
        agent = ProxyAgent(self.reactor, contextFactory=get_test_https_policy(), use_proxy=True)
        self._test_request_direct_connection(agent, b'https', b'test.com', b'abc')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888', 'no_proxy': '*'})
    def test_http_request_via_no_proxy_star(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        agent = ProxyAgent(self.reactor, use_proxy=True)
        self._test_request_direct_connection(agent, b'http', b'test.com', b'')

    @patch.dict(os.environ, {'https_proxy': 'proxy.com', 'no_proxy': '*'})
    def test_https_request_via_no_proxy_star(self) -> None:
        if False:
            print('Hello World!')
        agent = ProxyAgent(self.reactor, contextFactory=get_test_https_policy(), use_proxy=True)
        self._test_request_direct_connection(agent, b'https', b'test.com', b'abc')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888', 'no_proxy': 'unused.com'})
    def test_http_request_via_proxy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that requests can be made through a proxy.\n        '
        self._do_http_request_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=None)

    @patch.dict(os.environ, {'http_proxy': 'bob:pinkponies@proxy.com:8888', 'no_proxy': 'unused.com'})
    def test_http_request_via_proxy_with_auth(self) -> None:
        if False:
            return 10
        '\n        Tests that authenticated requests can be made through a proxy.\n        '
        self._do_http_request_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=b'bob:pinkponies')

    @patch.dict(os.environ, {'http_proxy': 'https://proxy.com:8888', 'no_proxy': 'unused.com'})
    def test_http_request_via_https_proxy(self) -> None:
        if False:
            while True:
                i = 10
        self._do_http_request_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=None)

    @patch.dict(os.environ, {'http_proxy': 'https://bob:pinkponies@proxy.com:8888', 'no_proxy': 'unused.com'})
    def test_http_request_via_https_proxy_with_auth(self) -> None:
        if False:
            while True:
                i = 10
        self._do_http_request_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=b'bob:pinkponies')

    @patch.dict(os.environ, {'https_proxy': 'proxy.com', 'no_proxy': 'unused.com'})
    def test_https_request_via_proxy(self) -> None:
        if False:
            return 10
        'Tests that TLS-encrypted requests can be made through a proxy'
        self._do_https_request_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=None)

    @patch.dict(os.environ, {'https_proxy': 'bob:pinkponies@proxy.com', 'no_proxy': 'unused.com'})
    def test_https_request_via_proxy_with_auth(self) -> None:
        if False:
            print('Hello World!')
        'Tests that authenticated, TLS-encrypted requests can be made through a proxy'
        self._do_https_request_via_proxy(expect_proxy_ssl=False, expected_auth_credentials=b'bob:pinkponies')

    @patch.dict(os.environ, {'https_proxy': 'https://proxy.com', 'no_proxy': 'unused.com'})
    def test_https_request_via_https_proxy(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that TLS-encrypted requests can be made through a proxy'
        self._do_https_request_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=None)

    @patch.dict(os.environ, {'https_proxy': 'https://bob:pinkponies@proxy.com', 'no_proxy': 'unused.com'})
    def test_https_request_via_https_proxy_with_auth(self) -> None:
        if False:
            print('Hello World!')
        'Tests that authenticated, TLS-encrypted requests can be made through a proxy'
        self._do_https_request_via_proxy(expect_proxy_ssl=True, expected_auth_credentials=b'bob:pinkponies')

    def _do_http_request_via_proxy(self, expect_proxy_ssl: bool=False, expected_auth_credentials: Optional[bytes]=None) -> None:
        if False:
            return 10
        'Send a http request via an agent and check that it is correctly received at\n            the proxy. The proxy can use either http or https.\n        Args:\n            expect_proxy_ssl: True if we expect the request to connect via https to proxy\n            expected_auth_credentials: credentials to authenticate at proxy\n        '
        if expect_proxy_ssl:
            agent = ProxyAgent(self.reactor, use_proxy=True, contextFactory=get_test_https_policy())
        else:
            agent = ProxyAgent(self.reactor, use_proxy=True)
        self.reactor.lookups['proxy.com'] = '1.2.3.5'
        d = agent.request(b'GET', b'http://test.com')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.5')
        self.assertEqual(port, 8888)
        http_server = self._make_connection(client_factory, _get_test_protocol_factory(), ssl=expect_proxy_ssl, tls_sanlist=[b'DNS:proxy.com'] if expect_proxy_ssl else None, expected_sni=b'proxy.com' if expect_proxy_ssl else None)
        assert isinstance(http_server, HTTPChannel)
        self.reactor.advance(0)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        proxy_auth_header_values = request.requestHeaders.getRawHeaders(b'Proxy-Authorization')
        if expected_auth_credentials is not None:
            encoded_credentials = base64.b64encode(expected_auth_credentials)
            expected_header_value = b'Basic ' + encoded_credentials
            self.assertIn(expected_header_value, proxy_auth_header_values)
        else:
            self.assertIsNone(proxy_auth_header_values)
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'http://test.com')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'test.com'])
        request.write(b'result')
        request.finish()
        self.reactor.advance(0)
        resp = self.successResultOf(d)
        body = self.successResultOf(treq.content(resp))
        self.assertEqual(body, b'result')

    def _do_https_request_via_proxy(self, expect_proxy_ssl: bool=False, expected_auth_credentials: Optional[bytes]=None) -> None:
        if False:
            print('Hello World!')
        'Send a https request via an agent and check that it is correctly received at\n            the proxy and client. The proxy can use either http or https.\n        Args:\n            expect_proxy_ssl: True if we expect the request to connect via https to proxy\n            expected_auth_credentials: credentials to authenticate at proxy\n        '
        agent = ProxyAgent(self.reactor, contextFactory=get_test_https_policy(), use_proxy=True)
        self.reactor.lookups['proxy.com'] = '1.2.3.5'
        d = agent.request(b'GET', b'https://test.com/abc')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.5')
        self.assertEqual(port, 1080)
        proxy_server = self._make_connection(client_factory, _get_test_protocol_factory(), ssl=expect_proxy_ssl, tls_sanlist=[b'DNS:proxy.com'] if expect_proxy_ssl else None, expected_sni=b'proxy.com' if expect_proxy_ssl else None)
        assert isinstance(proxy_server, HTTPChannel)
        self.assertEqual(len(proxy_server.requests), 1)
        request = proxy_server.requests[0]
        self.assertEqual(request.method, b'CONNECT')
        self.assertEqual(request.path, b'test.com:443')
        proxy_auth_header_values = request.requestHeaders.getRawHeaders(b'Proxy-Authorization')
        if expected_auth_credentials is not None:
            encoded_credentials = base64.b64encode(expected_auth_credentials)
            expected_header_value = b'Basic ' + encoded_credentials
            self.assertIn(expected_header_value, proxy_auth_header_values)
        else:
            self.assertIsNone(proxy_auth_header_values)
        proxy_server.persistent = True
        request.finish()
        server_ssl_protocol = wrap_server_factory_for_tls(_get_test_protocol_factory(), self.reactor, sanlist=[b'DNS:test.com']).buildProtocol(dummy_address)
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
        expected_sni = b'test.com'
        self.assertEqual(server_name, expected_sni, f'Expected SNI {expected_sni!s} but got {server_name!s}')
        http_server = server_ssl_protocol.wrappedProtocol
        assert isinstance(http_server, HTTPChannel)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/abc')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'test.com'])
        proxy_auth_header_values = request.requestHeaders.getRawHeaders(b'Proxy-Authorization')
        self.assertIsNone(proxy_auth_header_values)
        request.write(b'result')
        request.finish()
        self.reactor.advance(0)
        resp = self.successResultOf(d)
        body = self.successResultOf(treq.content(resp))
        self.assertEqual(body, b'result')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888'})
    def test_http_request_via_proxy_with_blocklist(self) -> None:
        if False:
            print('Hello World!')
        agent = ProxyAgent(BlocklistingReactorWrapper(self.reactor, ip_allowlist=None, ip_blocklist=IPSet(['1.0.0.0/8'])), self.reactor, use_proxy=True)
        self.reactor.lookups['proxy.com'] = '1.2.3.5'
        d = agent.request(b'GET', b'http://test.com')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.5')
        self.assertEqual(port, 8888)
        http_server = self._make_connection(client_factory, _get_test_protocol_factory())
        assert isinstance(http_server, HTTPChannel)
        self.reactor.advance(0)
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'http://test.com')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'test.com'])
        request.write(b'result')
        request.finish()
        self.reactor.advance(0)
        resp = self.successResultOf(d)
        body = self.successResultOf(treq.content(resp))
        self.assertEqual(body, b'result')

    @patch.dict(os.environ, {'HTTPS_PROXY': 'proxy.com'})
    def test_https_request_via_uppercase_proxy_with_blocklist(self) -> None:
        if False:
            return 10
        agent = ProxyAgent(BlocklistingReactorWrapper(self.reactor, ip_allowlist=None, ip_blocklist=IPSet(['1.0.0.0/8'])), self.reactor, contextFactory=get_test_https_policy(), use_proxy=True)
        self.reactor.lookups['proxy.com'] = '1.2.3.5'
        d = agent.request(b'GET', b'https://test.com/abc')
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.5')
        self.assertEqual(port, 1080)
        proxy_server = self._make_connection(client_factory, _get_test_protocol_factory())
        assert isinstance(proxy_server, HTTPChannel)
        s2c_transport = checked_cast(FakeTransport, proxy_server.transport)
        client_protocol = checked_cast(_WrappingProtocol, s2c_transport.other)
        c2s_transport = checked_cast(FakeTransport, client_protocol.transport)
        self.reactor.advance(0)
        self.assertEqual(len(proxy_server.requests), 1)
        request = proxy_server.requests[0]
        self.assertEqual(request.method, b'CONNECT')
        self.assertEqual(request.path, b'test.com:443')
        proxy_server.persistent = True
        request.finish()
        ssl_factory = wrap_server_factory_for_tls(_get_test_protocol_factory(), self.reactor, sanlist=[b'DNS:test.com'])
        ssl_protocol = ssl_factory.buildProtocol(dummy_address)
        assert isinstance(ssl_protocol, TLSMemoryBIOProtocol)
        http_server = ssl_protocol.wrappedProtocol
        assert isinstance(http_server, HTTPChannel)
        ssl_protocol.makeConnection(FakeTransport(client_protocol, self.reactor, ssl_protocol))
        c2s_transport.other = ssl_protocol
        self.reactor.advance(0)
        server_name = ssl_protocol._tlsConnection.get_servername()
        expected_sni = b'test.com'
        self.assertEqual(server_name, expected_sni, f'Expected SNI {expected_sni!s} but got {server_name!s}')
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, b'/abc')
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [b'test.com'])
        request.write(b'result')
        request.finish()
        self.reactor.advance(0)
        resp = self.successResultOf(d)
        body = self.successResultOf(treq.content(resp))
        self.assertEqual(body, b'result')

    @patch.dict(os.environ, {'http_proxy': 'proxy.com:8888'})
    def test_proxy_with_no_scheme(self) -> None:
        if False:
            while True:
                i = 10
        http_proxy_agent = ProxyAgent(self.reactor, use_proxy=True)
        proxy_ep = checked_cast(HostnameEndpoint, http_proxy_agent.http_proxy_endpoint)
        self.assertEqual(proxy_ep._hostStr, 'proxy.com')
        self.assertEqual(proxy_ep._port, 8888)

    @patch.dict(os.environ, {'http_proxy': 'socks://proxy.com:8888'})
    def test_proxy_with_unsupported_scheme(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            ProxyAgent(self.reactor, use_proxy=True)

    @patch.dict(os.environ, {'http_proxy': 'http://proxy.com:8888'})
    def test_proxy_with_http_scheme(self) -> None:
        if False:
            print('Hello World!')
        http_proxy_agent = ProxyAgent(self.reactor, use_proxy=True)
        proxy_ep = checked_cast(HostnameEndpoint, http_proxy_agent.http_proxy_endpoint)
        self.assertEqual(proxy_ep._hostStr, 'proxy.com')
        self.assertEqual(proxy_ep._port, 8888)

    @patch.dict(os.environ, {'http_proxy': 'https://proxy.com:8888'})
    def test_proxy_with_https_scheme(self) -> None:
        if False:
            while True:
                i = 10
        https_proxy_agent = ProxyAgent(self.reactor, use_proxy=True)
        proxy_ep = checked_cast(_WrapperEndpoint, https_proxy_agent.http_proxy_endpoint)
        self.assertEqual(proxy_ep._wrappedEndpoint._hostStr, 'proxy.com')
        self.assertEqual(proxy_ep._wrappedEndpoint._port, 8888)

def _get_test_protocol_factory() -> IProtocolFactory:
    if False:
        print('Hello World!')
    'Get a protocol Factory which will build an HTTPChannel\n\n    Returns:\n        interfaces.IProtocolFactory\n    '
    server_factory = Factory.forProtocol(HTTPChannel)
    server_factory.log = _log_request
    return server_factory

def _log_request(request: str) -> None:
    if False:
        return 10
    'Implements Factory.log, which is expected by Request.finish'
    logger.info(f'Completed request {request}')