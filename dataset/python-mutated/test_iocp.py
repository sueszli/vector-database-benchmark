"""
Tests for L{twisted.internet.iocpreactor}.
"""
import errno
import sys
import time
from array import array
from socket import AF_INET, AF_INET6, SOCK_STREAM, SOL_SOCKET, socket
from struct import pack
from unittest import skipIf
from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IPushProducer
from twisted.python.log import msg
from twisted.trial.unittest import TestCase
try:
    from twisted.internet.iocpreactor import iocpsupport as _iocp, tcp, udp
    from twisted.internet.iocpreactor.abstract import FileHandle
    from twisted.internet.iocpreactor.const import SO_UPDATE_ACCEPT_CONTEXT
    from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
    from twisted.internet.iocpreactor.reactor import EVENTS_PER_LOOP, KEY_NORMAL, IOCPReactor
except ImportError:
    if sys.platform == 'win32':
        raise
    skip = 'This test only applies to IOCPReactor'
try:
    socket(AF_INET6, SOCK_STREAM).close()
except OSError as e:
    ipv6Skip = True
    ipv6SkipReason = str(e)
else:
    ipv6Skip = False
    ipv6SkipReason = ''

class SupportTests(TestCase):
    """
    Tests for L{twisted.internet.iocpreactor.iocpsupport}, low-level reactor
    implementation helpers.
    """

    def _acceptAddressTest(self, family, localhost):
        if False:
            while True:
                i = 10
        '\n        Create a C{SOCK_STREAM} connection to localhost using a socket with an\n        address family of C{family} and assert that the result of\n        L{iocpsupport.get_accept_addrs} is consistent with the result of\n        C{socket.getsockname} and C{socket.getpeername}.\n\n        A port starts listening (is bound) at the low-level socket without\n        calling accept() yet.\n        A client is then connected.\n        After the client is connected IOCP accept() is called, which is the\n        target of these tests.\n\n        Most of the time, the socket is ready instantly, but sometimes\n        the socket is not ready right away after calling IOCP accept().\n        It should not take more than 5 seconds for a socket to be ready, as\n        the client connection is already made over the loopback interface.\n\n        These are flaky tests.\n        Tweak the failure rate by changing the number of retries and the\n        wait/sleep between retries.\n\n        If you will need to update the retries to wait more than 5 seconds\n        for the port to be available, then there might a bug in the code and\n        not the test (or a very, very busy VM running the tests).\n        '
        msg(f'family = {family!r}')
        port = socket(family, SOCK_STREAM)
        self.addCleanup(port.close)
        port.bind(('', 0))
        port.listen(1)
        client = socket(family, SOCK_STREAM)
        self.addCleanup(client.close)
        client.setblocking(False)
        try:
            client.connect((localhost, port.getsockname()[1]))
        except OSError as e:
            self.assertIn(e.errno, (errno.EINPROGRESS, errno.EWOULDBLOCK))
        server = socket(family, SOCK_STREAM)
        self.addCleanup(server.close)
        buff = array('B', b'\x00' * 256)
        self.assertEqual(0, _iocp.accept(port.fileno(), server.fileno(), buff, None))
        for attemptsRemaining in reversed(range(5)):
            try:
                server.setsockopt(SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT, pack('P', port.fileno()))
                break
            except OSError as socketError:
                if socketError.errno != getattr(errno, 'WSAENOTCONN'):
                    raise
                if attemptsRemaining == 0:
                    raise
            time.sleep(0.2)
        self.assertEqual((family, client.getpeername()[:2], client.getsockname()[:2]), _iocp.get_accept_addrs(server.fileno(), buff))

    def test_ipv4AcceptAddress(self):
        if False:
            return 10
        '\n        L{iocpsupport.get_accept_addrs} returns a three-tuple of address\n        information about the socket associated with the file descriptor passed\n        to it.  For a connection using IPv4:\n\n          - the first element is C{AF_INET}\n          - the second element is a two-tuple of a dotted decimal notation IPv4\n            address and a port number giving the peer address of the connection\n          - the third element is the same type giving the host address of the\n            connection\n        '
        self._acceptAddressTest(AF_INET, '127.0.0.1')

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_ipv6AcceptAddress(self):
        if False:
            print('Hello World!')
        '\n        Like L{test_ipv4AcceptAddress}, but for IPv6 connections.\n        In this case:\n\n          - the first element is C{AF_INET6}\n          - the second element is a two-tuple of a hexadecimal IPv6 address\n            literal and a port number giving the peer address of the connection\n          - the third element is the same type giving the host address of the\n            connection\n        '
        self._acceptAddressTest(AF_INET6, '::1')

class IOCPReactorTests(TestCase):

    def test_noPendingTimerEvents(self):
        if False:
            print('Hello World!')
        '\n        Test reactor behavior (doIteration) when there are no pending time\n        events.\n        '
        ir = IOCPReactor()
        ir.wakeUp()
        self.assertFalse(ir.doIteration(None))

    def test_reactorInterfaces(self):
        if False:
            while True:
                i = 10
        '\n        Verify that IOCP socket-representing classes implement IReadWriteHandle\n        '
        self.assertTrue(verifyClass(IReadWriteHandle, tcp.Connection))
        self.assertTrue(verifyClass(IReadWriteHandle, udp.Port))

    def test_fileHandleInterfaces(self):
        if False:
            return 10
        '\n        Verify that L{Filehandle} implements L{IPushProducer}.\n        '
        self.assertTrue(verifyClass(IPushProducer, FileHandle))

    def test_maxEventsPerIteration(self):
        if False:
            return 10
        "\n        Verify that we don't lose an event when more than EVENTS_PER_LOOP\n        events occur in the same reactor iteration\n        "

        class FakeFD:
            counter = 0

            def logPrefix(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'FakeFD'

            def cb(self, rc, bytes, evt):
                if False:
                    return 10
                self.counter += 1
        ir = IOCPReactor()
        fd = FakeFD()
        event = _iocp.Event(fd.cb, fd)
        for _ in range(EVENTS_PER_LOOP + 1):
            ir.port.postEvent(0, KEY_NORMAL, event)
        ir.doIteration(None)
        self.assertEqual(fd.counter, EVENTS_PER_LOOP)
        ir.doIteration(0)
        self.assertEqual(fd.counter, EVENTS_PER_LOOP + 1)