from typing import Any, Dict, Generator
from unittest.mock import ANY, Mock, create_autospec
from netaddr import IPSet
from parameterized import parameterized
from twisted.internet import defer
from twisted.internet.defer import Deferred, TimeoutError
from twisted.internet.error import ConnectingCancelledError, DNSLookupError
from twisted.test.proto_helpers import MemoryReactor, StringTransport
from twisted.web.client import Agent, ResponseNeverReceived
from twisted.web.http import HTTPChannel
from twisted.web.http_headers import Headers
from synapse.api.errors import HttpResponseException, RequestSendFailed
from synapse.config._base import ConfigError
from synapse.http.matrixfederationclient import ByteParser, MatrixFederationHttpClient, MatrixFederationRequest
from synapse.logging.context import SENTINEL_CONTEXT, LoggingContext, LoggingContextOrSentinel, current_context
from synapse.server import HomeServer
from synapse.util import Clock
from tests.replication._base import BaseMultiWorkerStreamTestCase
from tests.server import FakeTransport
from tests.test_utils import FakeResponse
from tests.unittest import HomeserverTestCase, override_config

def check_logcontext(context: LoggingContextOrSentinel) -> None:
    if False:
        i = 10
        return i + 15
    current = current_context()
    if current is not context:
        raise AssertionError('Expected logcontext %s but was %s' % (context, current))

class FederationClientTests(HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        hs = self.setup_test_homeserver(reactor=reactor, clock=clock)
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.cl = MatrixFederationHttpClient(self.hs, None)
        self.reactor.lookups['testserv'] = '1.2.3.4'

    def test_client_get(self) -> None:
        if False:
            print('Hello World!')
        '\n        happy-path test of a GET request\n        '

        @defer.inlineCallbacks
        def do_request() -> Generator['Deferred[Any]', object, object]:
            if False:
                return 10
            with LoggingContext('one') as context:
                fetch_d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar'))
                self.assertNoResult(fetch_d)
                check_logcontext(SENTINEL_CONTEXT)
                try:
                    fetch_res = (yield fetch_d)
                    return fetch_res
                finally:
                    check_logcontext(context)
        test_d = do_request()
        self.pump()
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8008)
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self.assertRegex(transport.value(), b'^GET /foo/bar')
        self.assertRegex(transport.value(), b'Host: testserv:8008')
        self.assertNoResult(test_d)
        res_json = b'{ "a": 1 }'
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nServer: Fake\r\nContent-Type: application/json\r\nContent-Length: %i\r\n\r\n%s' % (len(res_json), res_json))
        self.pump()
        res = self.successResultOf(test_d)
        self.assertEqual(res, {'a': 1})

    def test_dns_error(self) -> None:
        if False:
            print('Hello World!')
        '\n        If the DNS lookup returns an error, it will bubble up.\n        '
        d = defer.ensureDeferred(self.cl.get_json('testserv2:8008', 'foo/bar', timeout=10000))
        self.pump()
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, DNSLookupError)

    def test_client_connection_refused(self) -> None:
        if False:
            i = 10
            return i + 15
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar', timeout=10000))
        self.pump()
        self.assertNoResult(d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8008)
        e = Exception('go away')
        factory.clientConnectionFailed(None, e)
        self.pump(0.5)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertIs(f.value.inner_exception, e)

    def test_client_never_connect(self) -> None:
        if False:
            return 10
        "\n        If the HTTP request is not connected and is timed out, it'll give a\n        ConnectingCancelledError or TimeoutError.\n        "
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar', timeout=10000))
        self.pump()
        self.assertNoResult(d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0][0], '1.2.3.4')
        self.assertEqual(clients[0][1], 8008)
        self.assertNoResult(d)
        self.reactor.advance(10.5)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, (ConnectingCancelledError, TimeoutError))

    def test_client_connect_no_response(self) -> None:
        if False:
            while True:
                i = 10
        "\n        If the HTTP request is connected, but gets no response before being\n        timed out, it'll give a ResponseNeverReceived.\n        "
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar', timeout=10000))
        self.pump()
        self.assertNoResult(d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0][0], '1.2.3.4')
        self.assertEqual(clients[0][1], 8008)
        conn = Mock()
        client = clients[0][2].buildProtocol(None)
        client.makeConnection(conn)
        self.assertNoResult(d)
        self.reactor.advance(10.5)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, ResponseNeverReceived)

    def test_client_ip_range_blocklist(self) -> None:
        if False:
            print('Hello World!')
        'Ensure that Synapse does not try to connect to blocked IPs'
        self.hs.config.server.federation_ip_range_blocklist = IPSet(['127.0.0.0/8', 'fe80::/64'])
        self.reactor.lookups['internal'] = '127.0.0.1'
        self.reactor.lookups['internalv6'] = 'fe80:0:0:0:0:8a2e:370:7337'
        self.reactor.lookups['fine'] = '10.20.30.40'
        cl = MatrixFederationHttpClient(self.hs, None)
        d = defer.ensureDeferred(cl.get_json('internal:8008', 'foo/bar', timeout=10000))
        self.assertNoResult(d)
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 0)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, DNSLookupError)
        d = defer.ensureDeferred(cl.post_json('internalv6:8008', 'foo/bar', timeout=10000))
        self.assertNoResult(d)
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 0)
        f = self.failureResultOf(d, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, DNSLookupError)
        d = defer.ensureDeferred(cl.post_json('fine:8008', 'foo/bar', timeout=10000))
        self.assertNoResult(d)
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertNotEqual(len(clients), 0)
        f = self.failureResultOf(d, RequestSendFailed)
        self.assertIsInstance(f.value.inner_exception, ConnectingCancelledError)

    def test_client_gets_headers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Once the client gets the headers, _request returns successfully.\n        '
        request = MatrixFederationRequest(method='GET', destination='testserv:8008', path='foo/bar')
        d = defer.ensureDeferred(self.cl._send_request(request, timeout=10000))
        self.pump()
        conn = Mock()
        clients = self.reactor.tcpClients
        client = clients[0][2].buildProtocol(None)
        client.makeConnection(conn)
        self.assertNoResult(d)
        client.dataReceived(b'HTTP/1.1 200 OK\r\nServer: Fake\r\n\r\n')
        r = self.successResultOf(d)
        self.assertEqual(r.code, 200)

    @parameterized.expand(['get_json', 'post_json', 'delete_json', 'put_json'])
    def test_timeout_reading_body(self, method_name: str) -> None:
        if False:
            return 10
        "\n        If the HTTP request is connected, but gets no response before being\n        timed out, it'll give a RequestSendFailed with can_retry.\n        "
        method = getattr(self.cl, method_name)
        d = defer.ensureDeferred(method('testserv:8008', 'foo/bar', timeout=10000))
        self.pump()
        conn = Mock()
        clients = self.reactor.tcpClients
        client = clients[0][2].buildProtocol(None)
        client.makeConnection(conn)
        self.assertNoResult(d)
        client.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nServer: Fake\r\n\r\n')
        self.reactor.advance(10.5)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertTrue(f.value.can_retry)
        self.assertIsInstance(f.value.inner_exception, defer.TimeoutError)

    def test_client_requires_trailing_slashes(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        If a connection is made to a client but the client rejects it due to\n        requiring a trailing slash. We need to retry the request with a\n        trailing slash. Workaround for Synapse <= v0.99.3, explained in\n        https://github.com/matrix-org/synapse/issues/3622.\n        '
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar', try_trailing_slash_on_400=True))
        self.pump()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (_host, _port, factory, _timeout, _bindAddress) = clients[0]
        client = factory.buildProtocol(None)
        conn = StringTransport()
        client.makeConnection(conn)
        self.assertRegex(conn.value(), b'^GET /foo/bar')
        conn.clear()
        client.dataReceived(b'HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 59\r\n\r\n{"errcode":"M_UNRECOGNIZED","error":"Unrecognized request"}')
        self.assertRegex(conn.value(), b'^GET /foo/bar/')
        client.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}')
        r = self.successResultOf(d)
        self.assertEqual(r, {})

    def test_client_does_not_retry_on_400_plus(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Another test for trailing slashes but now test that we don't retry on\n        trailing slashes on a non-400/M_UNRECOGNIZED response.\n\n        See test_client_requires_trailing_slashes() for context.\n        "
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar', try_trailing_slash_on_400=True))
        self.pump()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (_host, _port, factory, _timeout, _bindAddress) = clients[0]
        client = factory.buildProtocol(None)
        conn = StringTransport()
        client.makeConnection(conn)
        self.assertRegex(conn.value(), b'^GET /foo/bar')
        conn.clear()
        client.dataReceived(b'HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}')
        self.assertEqual(conn.value(), b'')
        self.failureResultOf(d)

    def test_client_sends_body(self) -> None:
        if False:
            return 10
        defer.ensureDeferred(self.cl.post_json('testserv:8008', 'foo/bar', timeout=10000, data={'a': 'b'}))
        self.pump()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        client = clients[0][2].buildProtocol(None)
        server = HTTPChannel()
        client.makeConnection(FakeTransport(server, self.reactor))
        server.makeConnection(FakeTransport(client, self.reactor))
        self.pump(0.1)
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        content = request.content.read()
        self.assertEqual(content, b'{"a":"b"}')

    def test_closes_connection(self) -> None:
        if False:
            i = 10
            return i + 15
        'Check that the client closes unused HTTP connections'
        d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar'))
        self.pump()
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (_host, _port, factory, _timeout, _bindAddress) = clients[0]
        client = factory.buildProtocol(None)
        conn = StringTransport()
        client.makeConnection(conn)
        self.assertRegex(conn.value(), b'^GET /foo/bar')
        client.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}')
        r = self.successResultOf(d)
        self.assertEqual(r, {})
        self.assertFalse(conn.disconnecting)
        self.reactor.advance(120)
        self.assertTrue(conn.disconnecting)

    @parameterized.expand([(b'',), (b'foo',), (b'{"a": Infinity}',)])
    def test_json_error(self, return_value: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test what happens if invalid JSON is returned from the remote endpoint.\n        '
        test_d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar'))
        self.pump()
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8008)
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self.assertRegex(transport.value(), b'^GET /foo/bar')
        self.assertRegex(transport.value(), b'Host: testserv:8008')
        self.assertNoResult(test_d)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nServer: Fake\r\nContent-Type: application/json\r\nContent-Length: %i\r\n\r\n%s' % (len(return_value), return_value))
        self.pump()
        f = self.failureResultOf(test_d)
        self.assertIsInstance(f.value, RequestSendFailed)

    def test_too_big(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test what happens if a huge response is returned from the remote endpoint.\n        '
        test_d = defer.ensureDeferred(self.cl.get_json('testserv:8008', 'foo/bar'))
        self.pump()
        self.assertNoResult(test_d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        (host, port, factory, _timeout, _bindAddress) = clients[0]
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 8008)
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self.assertRegex(transport.value(), b'^GET /foo/bar')
        self.assertRegex(transport.value(), b'Host: testserv:8008')
        self.assertNoResult(test_d)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nServer: Fake\r\nContent-Type: application/json\r\n\r\n')
        self.pump()
        self.assertNoResult(test_d)
        sent = 0
        chunk_size = 1024 * 512
        while not test_d.called:
            protocol.dataReceived(b'a' * chunk_size)
            sent += chunk_size
            self.assertLessEqual(sent, ByteParser.MAX_RESPONSE_SIZE)
        self.assertEqual(sent, ByteParser.MAX_RESPONSE_SIZE)
        f = self.failureResultOf(test_d)
        self.assertIsInstance(f.value, RequestSendFailed)
        self.assertTrue(transport.disconnecting)

    def test_build_auth_headers_rejects_falsey_destinations(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self.cl.build_auth_headers(None, b'GET', b'https://example.com')
        with self.assertRaises(ValueError):
            self.cl.build_auth_headers(b'', b'GET', b'https://example.com')
        with self.assertRaises(ValueError):
            self.cl.build_auth_headers(None, b'GET', b'https://example.com', destination_is=b'')
        with self.assertRaises(ValueError):
            self.cl.build_auth_headers(b'', b'GET', b'https://example.com', destination_is=b'')

    @override_config({'federation': {'client_timeout': '180s', 'max_long_retry_delay': '100s', 'max_short_retry_delay': '7s', 'max_long_retries': 20, 'max_short_retries': 5}})
    def test_configurable_retry_and_delay_values(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.cl.default_timeout_seconds, 180)
        self.assertEqual(self.cl.max_long_retry_delay_seconds, 100)
        self.assertEqual(self.cl.max_short_retry_delay_seconds, 7)
        self.assertEqual(self.cl.max_long_retries, 20)
        self.assertEqual(self.cl.max_short_retries, 5)

class FederationClientProxyTests(BaseMultiWorkerStreamTestCase):

    def default_config(self) -> Dict[str, Any]:
        if False:
            return 10
        conf = super().default_config()
        conf['instance_map'] = {'main': {'host': 'testserv', 'port': 8765}, 'federation_sender': {'host': 'testserv', 'port': 1001}}
        return conf

    @override_config({'outbound_federation_restricted_to': ['federation_sender'], 'worker_replication_secret': 'secret'})
    def test_proxy_requests_through_federation_sender_worker(self) -> None:
        if False:
            return 10
        '\n        Test that all outbound federation requests go through the `federation_sender`\n        worker\n        '
        mock_client_on_federation_sender = Mock()
        mock_agent_on_federation_sender = create_autospec(Agent, spec_set=True)
        mock_client_on_federation_sender.agent = mock_agent_on_federation_sender
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'federation_sender'}, federation_http_client=mock_client_on_federation_sender)
        mock_agent_on_federation_sender.request.side_effect = lambda *args, **kwargs: defer.succeed(FakeResponse.json(payload={'foo': 'bar'}))
        test_request_from_main_process_d = defer.ensureDeferred(self.hs.get_federation_http_client().get_json('remoteserv:8008', 'foo/bar'))
        self.pump()
        mock_agent_on_federation_sender.request.assert_called_once_with(b'GET', b'matrix-federation://remoteserv:8008/foo/bar', headers=ANY, bodyProducer=ANY)
        res = self.successResultOf(test_request_from_main_process_d)
        self.assertEqual(res, {'foo': 'bar'})

    @override_config({'outbound_federation_restricted_to': ['federation_sender'], 'worker_replication_secret': 'secret'})
    def test_proxy_request_with_network_error_through_federation_sender_worker(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that when the outbound federation request fails with a network related\n        error, a sensible error makes its way back to the main process.\n        '
        mock_client_on_federation_sender = Mock()
        mock_agent_on_federation_sender = create_autospec(Agent, spec_set=True)
        mock_client_on_federation_sender.agent = mock_agent_on_federation_sender
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'federation_sender'}, federation_http_client=mock_client_on_federation_sender)
        mock_agent_on_federation_sender.request.side_effect = lambda *args, **kwargs: defer.fail(ResponseNeverReceived('fake error'))
        test_request_from_main_process_d = defer.ensureDeferred(self.hs.get_federation_http_client().get_json('remoteserv:8008', 'foo/bar'))
        self.pump(0.1)
        mock_agent_on_federation_sender.request.assert_called_with(b'GET', b'matrix-federation://remoteserv:8008/foo/bar', headers=ANY, bodyProducer=ANY)
        failure_res = self.failureResultOf(test_request_from_main_process_d)
        self.assertIsInstance(failure_res.value, RequestSendFailed)
        self.assertIsInstance(failure_res.value.inner_exception, HttpResponseException)
        self.assertEqual(failure_res.value.inner_exception.code, 502)

    @override_config({'outbound_federation_restricted_to': ['federation_sender'], 'worker_replication_secret': 'secret'})
    def test_proxy_requests_and_discards_hop_by_hop_headers(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test to make sure hop-by-hop headers and addional headers defined in the\n        `Connection` header are discarded when proxying requests\n        '
        mock_client_on_federation_sender = Mock()
        mock_agent_on_federation_sender = create_autospec(Agent, spec_set=True)
        mock_client_on_federation_sender.agent = mock_agent_on_federation_sender
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'federation_sender'}, federation_http_client=mock_client_on_federation_sender)
        mock_agent_on_federation_sender.request.side_effect = lambda *args, **kwargs: defer.succeed(FakeResponse(code=200, body=b'{"foo": "bar"}', headers=Headers({'Content-Type': ['application/json'], 'Connection': ['close, X-Foo, X-Bar'], 'X-Foo': ['foo'], 'X-Bar': ['bar'], 'Proxy-Authorization': 'abcdef'})))
        test_request_from_main_process_d = defer.ensureDeferred(self.hs.get_federation_http_client().get_json_with_headers('remoteserv:8008', 'foo/bar'))
        self.pump()
        mock_agent_on_federation_sender.request.assert_called_once_with(b'GET', b'matrix-federation://remoteserv:8008/foo/bar', headers=ANY, bodyProducer=ANY)
        (res, headers) = self.successResultOf(test_request_from_main_process_d)
        header_names = set(headers.keys())
        self.assertNotIn(b'X-Foo', header_names)
        self.assertNotIn(b'X-Bar', header_names)
        self.assertNotIn(b'Proxy-Authorization', header_names)
        self.assertEqual(res, {'foo': 'bar'})

    @override_config({'outbound_federation_restricted_to': ['federation_sender'], 'worker_replication_secret': 'secret'})
    def test_not_able_to_proxy_requests_through_federation_sender_worker_when_no_secret_configured(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that we aren't able to proxy any outbound federation requests when\n        `worker_replication_secret` is not configured.\n        "
        with self.assertRaises(ConfigError):
            self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'federation_sender', 'worker_replication_secret': None})

    @override_config({'outbound_federation_restricted_to': ['federation_sender'], 'worker_replication_secret': 'secret'})
    def test_not_able_to_proxy_requests_through_federation_sender_worker_when_wrong_auth_given(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that we aren't able to proxy any outbound federation requests when the\n        wrong authorization is given.\n        "
        mock_client_on_federation_sender = Mock()
        mock_agent_on_federation_sender = create_autospec(Agent, spec_set=True)
        mock_client_on_federation_sender.agent = mock_agent_on_federation_sender
        self.make_worker_hs('synapse.app.generic_worker', {'worker_name': 'federation_sender', 'worker_replication_secret': 'wrong'}, federation_http_client=mock_client_on_federation_sender)
        test_request_from_main_process_d = defer.ensureDeferred(self.hs.get_federation_http_client().get_json('remoteserv:8008', 'foo/bar'))
        self.pump(0.1)
        mock_agent_on_federation_sender.request.assert_not_called()
        failure_res = self.failureResultOf(test_request_from_main_process_d)
        self.assertIsInstance(failure_res.value, HttpResponseException)
        self.assertEqual(failure_res.value.code, 401)