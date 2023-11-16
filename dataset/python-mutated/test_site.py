from twisted.internet.address import IPv6Address
from twisted.test.proto_helpers import MemoryReactor, StringTransport
from synapse.app.homeserver import SynapseHomeServer
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class SynapseRequestTestCase(HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        return self.setup_test_homeserver(homeserver_to_use=SynapseHomeServer)

    def test_large_request(self) -> None:
        if False:
            print('Hello World!')
        'overlarge HTTP requests should be rejected'
        self.hs.start_listening()
        (port, factory, _backlog, interface) = self.reactor.tcpServers[0]
        self.assertEqual(interface, '::')
        self.assertEqual(port, 0)
        client_address = IPv6Address('TCP', '::1', 2345)
        protocol = factory.buildProtocol(client_address)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.1\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n')
        while not transport.disconnecting:
            self.reactor.advance(1)
        self.assertRegex(transport.value().decode(), '^HTTP/1\\.1 404 ')
        protocol = factory.buildProtocol(client_address)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'POST / HTTP/1.1\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n')
        protocol.dataReceived(b'10000000\r\n')
        sent = 0
        while not transport.disconnected:
            self.assertLess(sent, 268435456, 'connection did not drop')
            protocol.dataReceived(b'\x00' * 1024)
            sent += 1024
        self.assertEqual(sent, 50 * 1024 * 1024 + 1024)