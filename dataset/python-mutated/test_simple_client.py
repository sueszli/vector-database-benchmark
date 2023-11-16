from unittest.mock import Mock
from netaddr import IPSet
from twisted.internet import defer
from twisted.internet.error import DNSLookupError
from twisted.test.proto_helpers import MemoryReactor
from synapse.http import RequestTimedOutError
from synapse.http.client import SimpleHttpClient
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class SimpleHttpClientTests(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: 'HomeServer') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.reactor.lookups['testserv'] = '1.2.3.4'
        self.cl = hs.get_simple_http_client()

    def test_dns_error(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        If the DNS lookup returns an error, it will bubble up.\n        '
        d = defer.ensureDeferred(self.cl.get_json('http://testserv2:8008/foo/bar'))
        self.pump()
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, DNSLookupError)

    def test_client_connection_refused(self) -> None:
        if False:
            while True:
                i = 10
        d = defer.ensureDeferred(self.cl.get_json('http://testserv:8008/foo/bar'))
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
        self.assertIs(f.value, e)

    def test_client_never_connect(self) -> None:
        if False:
            print('Hello World!')
        "\n        If the HTTP request is not connected and is timed out, it'll give a\n        ConnectingCancelledError or TimeoutError.\n        "
        d = defer.ensureDeferred(self.cl.get_json('http://testserv:8008/foo/bar'))
        self.pump()
        self.assertNoResult(d)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0][0], '1.2.3.4')
        self.assertEqual(clients[0][1], 8008)
        self.assertNoResult(d)
        self.reactor.advance(120)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestTimedOutError)

    def test_client_connect_no_response(self) -> None:
        if False:
            return 10
        "\n        If the HTTP request is connected, but gets no response before being\n        timed out, it'll give a ResponseNeverReceived.\n        "
        d = defer.ensureDeferred(self.cl.get_json('http://testserv:8008/foo/bar'))
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
        self.reactor.advance(120)
        f = self.failureResultOf(d)
        self.assertIsInstance(f.value, RequestTimedOutError)

    def test_client_ip_range_blocklist(self) -> None:
        if False:
            i = 10
            return i + 15
        'Ensure that Synapse does not try to connect to blocked IPs'
        self.reactor.lookups['internal'] = '127.0.0.1'
        self.reactor.lookups['internalv6'] = 'fe80:0:0:0:0:8a2e:370:7337'
        ip_blocklist = IPSet(['127.0.0.0/8', 'fe80::/64'])
        cl = SimpleHttpClient(self.hs, ip_blocklist=ip_blocklist)
        d = defer.ensureDeferred(cl.get_json('http://internal:8008/foo/bar'))
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 0)
        self.failureResultOf(d, DNSLookupError)
        d = defer.ensureDeferred(cl.post_json_get_json('http://internalv6:8008/foo/bar', {}))
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertEqual(len(clients), 0)
        self.failureResultOf(d, DNSLookupError)
        d = defer.ensureDeferred(cl.get_json('http://testserv:8008/foo/bar'))
        self.assertNoResult(d)
        self.pump(1)
        clients = self.reactor.tcpClients
        self.assertNotEqual(len(clients), 0)
        self.failureResultOf(d, RequestTimedOutError)