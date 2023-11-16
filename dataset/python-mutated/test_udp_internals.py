"""
Tests for the internal implementation details of L{twisted.internet.udp}.
"""
from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
if platformType == 'win32':
    from errno import WSAEWOULDBLOCK as EWOULDBLOCK
else:
    from errno import EWOULDBLOCK

class StringUDPSocket:
    """
    A fake UDP socket object, which returns a fixed sequence of strings and/or
    socket errors.  Useful for testing.

    @ivar retvals: A C{list} containing either strings or C{socket.error}s.

    @ivar connectedAddr: The address the socket is connected to.
    """

    def __init__(self, retvals: list[bytes | socket.error]) -> None:
        if False:
            print('Hello World!')
        self.retvals = retvals
        self.connectedAddr: object | None = None

    def connect(self, addr: object) -> None:
        if False:
            i = 10
            return i + 15
        self.connectedAddr = addr

    def recvfrom(self, size: int) -> tuple[bytes, None]:
        if False:
            i = 10
            return i + 15
        '\n        Return (or raise) the next value from C{self.retvals}.\n        '
        ret = self.retvals.pop(0)
        if isinstance(ret, socket.error):
            raise ret
        return (ret, None)

class KeepReads(DatagramProtocol):
    """
    Accumulate reads in a list.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.reads: list[bytes] = []

    def datagramReceived(self, data: bytes, addr: object) -> None:
        if False:
            print('Hello World!')
        self.reads.append(data)

class ErrorsTests(unittest.SynchronousTestCase):
    """
    Error handling tests for C{udp.Port}.
    """

    def test_socketReadNormal(self) -> None:
        if False:
            return 10
        '\n        Socket reads with some good data followed by a socket error which can\n        be ignored causes reading to stop, and no log messages to be logged.\n        '
        udp._sockErrReadIgnore.append(-7000)
        self.addCleanup(udp._sockErrReadIgnore.remove, -7000)
        protocol = KeepReads()
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'result', b'123', socket.error(-7000), b'456', socket.error(-7000)])
        port.doRead()
        self.assertEqual(protocol.reads, [b'result', b'123'])
        port.doRead()
        self.assertEqual(protocol.reads, [b'result', b'123', b'456'])

    def test_readImmediateError(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        If the socket is unconnected, socket reads with an immediate\n        connection refusal are ignored, and reading stops. The protocol's\n        C{connectionRefused} method is not called.\n        "
        udp._sockErrReadRefuse.append(-6000)
        self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
        protocol = KeepReads()
        protocol.connectionRefused = lambda : 1 / 0
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a'])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a', b'b'])

    def test_connectedReadImmediateError(self) -> None:
        if False:
            while True:
                i = 10
        "\n        If the socket connected, socket reads with an immediate\n        connection refusal are ignored, and reading stops. The protocol's\n        C{connectionRefused} method is called.\n        "
        udp._sockErrReadRefuse.append(-6000)
        self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
        protocol = KeepReads()
        refused = []
        protocol.connectionRefused = lambda : refused.append(True)
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
        port.connect('127.0.0.1', 9999)
        port.doRead()
        self.assertEqual(protocol.reads, [b'a'])
        self.assertEqual(refused, [True])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a', b'b'])
        self.assertEqual(refused, [True])

    def test_readUnknownError(self) -> None:
        if False:
            print('Hello World!')
        '\n        Socket reads with an unknown socket error are raised.\n        '
        protocol = KeepReads()
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'good', socket.error(-1337)])
        self.assertRaises(socket.error, port.doRead)
        self.assertEqual(protocol.reads, [b'good'])