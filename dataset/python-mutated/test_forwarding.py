"""
Tests for L{twisted.conch.ssh.forwarding}.
"""
from twisted.python.reflect import requireModule
cryptography = requireModule('cryptography')
if cryptography:
    from twisted.conch.ssh import forwarding
from twisted.internet.address import IPv6Address
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.trial import unittest

class TestSSHConnectForwardingChannel(unittest.TestCase):
    """
    Unit and integration tests for L{SSHConnectForwardingChannel}.
    """
    if not cryptography:
        skip = 'Cannot run without cryptography'

    def makeTCPConnection(self, reactor: MemoryReactorClock) -> None:
        if False:
            while True:
                i = 10
        '\n        Fake that connection was established for first connectTCP request made\n        on C{reactor}.\n\n        @param reactor: Reactor on which to fake the connection.\n        @type  reactor: A reactor.\n        '
        factory = reactor.tcpClients[0][2]
        connector = reactor.connectors[0]
        protocol = factory.buildProtocol(None)
        transport = StringTransport(peerAddress=connector.getDestination())
        protocol.makeConnection(transport)

    def test_channelOpenHostnameRequests(self) -> None:
        if False:
            return 10
        "\n        When a hostname is sent as part of forwarding requests, it\n        is resolved using HostnameEndpoint's resolver.\n        "
        sut = forwarding.SSHConnectForwardingChannel(hostport=('fwd.example.org', 1234))
        memoryReactor = MemoryReactorClock()
        sut._reactor = deterministicResolvingReactor(memoryReactor, ['::1'])
        sut.channelOpen(None)
        self.makeTCPConnection(memoryReactor)
        self.successResultOf(sut._channelOpenDeferred)
        self.assertIsInstance(sut.client, forwarding.SSHForwardingClient)
        self.assertEqual(IPv6Address('TCP', '::1', 1234), sut.client.transport.getPeer())