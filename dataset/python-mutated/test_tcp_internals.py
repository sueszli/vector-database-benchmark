"""
Whitebox tests for TCP APIs.
"""
import errno
import os
import socket
try:
    import resource
except ImportError:
    resource = None
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import _ACCEPT_ERRORS, EAGAIN, ECONNABORTED, EINPROGRESS, EMFILE, ENFILE, ENOBUFS, ENOMEM, EPERM, EWOULDBLOCK, Port
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase

@skipIf(not interfaces.IReactorFDSet.providedBy(reactor), 'This test only applies to reactors that implement IReactorFDset')
class PlatformAssumptionsTests(TestCase):
    """
    Test assumptions about platform behaviors.
    """
    socketLimit = 8192

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.openSockets = []
        if resource is not None:
            from twisted.internet.process import _listOpenFDs
            newLimit = len(_listOpenFDs()) + 2
            self.originalFileLimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (newLimit, self.originalFileLimit[1]))
            self.socketLimit = newLimit + 100

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        while self.openSockets:
            self.openSockets.pop().close()
        if resource is not None:
            currentHardLimit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            newSoftLimit = min(self.originalFileLimit[0], currentHardLimit)
            resource.setrlimit(resource.RLIMIT_NOFILE, (newSoftLimit, currentHardLimit))

    def socket(self):
        if False:
            while True:
                i = 10
        '\n        Create and return a new socket object, also tracking it so it can be\n        closed in the test tear down.\n        '
        s = socket.socket()
        self.openSockets.append(s)
        return s

    @skipIf(platform.getType() == 'win32', 'Windows requires an unacceptably large amount of resources to provoke this behavior in the naive manner.')
    def test_acceptOutOfFiles(self):
        if False:
            print('Hello World!')
        '\n        Test that the platform accept(2) call fails with either L{EMFILE} or\n        L{ENOBUFS} when there are too many file descriptors open.\n        '
        port = self.socket()
        port.bind(('127.0.0.1', 0))
        serverPortNumber = port.getsockname()[1]
        port.listen(5)
        client = self.socket()
        client.setblocking(False)
        for i in range(self.socketLimit):
            try:
                self.socket()
            except OSError as e:
                if e.args[0] in (EMFILE, ENOBUFS):
                    break
                else:
                    raise
        else:
            self.fail('Could provoke neither EMFILE nor ENOBUFS from platform.')
        self.assertIn(client.connect_ex(('127.0.0.1', serverPortNumber)), (0, EINPROGRESS))
        exc = self.assertRaises(socket.error, port.accept)
        self.assertIn(exc.args[0], (EMFILE, ENOBUFS))

@skipIf(not interfaces.IReactorFDSet.providedBy(reactor), 'This test only applies to reactors that implement IReactorFDset')
class SelectReactorTests(TestCase):
    """
    Tests for select-specific failure conditions.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.ports = []
        self.messages = []
        log.addObserver(self.messages.append)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        log.removeObserver(self.messages.append)
        return gatherResults([maybeDeferred(p.stopListening) for p in self.ports])

    def port(self, portNumber, factory, interface):
        if False:
            while True:
                i = 10
        '\n        Create, start, and return a new L{Port}, also tracking it so it can\n        be stopped in the test tear down.\n        '
        p = Port(portNumber, factory, interface=interface)
        p.startListening()
        self.ports.append(p)
        return p

    def _acceptFailureTest(self, socketErrorNumber):
        if False:
            while True:
                i = 10
        '\n        Test behavior in the face of an exception from C{accept(2)}.\n\n        On any exception which indicates the platform is unable or unwilling\n        to allocate further resources to us, the existing port should remain\n        listening, a message should be logged, and the exception should not\n        propagate outward from doRead.\n\n        @param socketErrorNumber: The errno to simulate from accept.\n        '

        class FakeSocket:
            """
            Pretend to be a socket in an overloaded system.
            """

            def accept(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise OSError(socketErrorNumber, os.strerror(socketErrorNumber))
        factory = ServerFactory()
        port = self.port(0, factory, interface='127.0.0.1')
        self.patch(port, 'socket', FakeSocket())
        port.doRead()
        expectedFormat = 'Could not accept new connection ({acceptError})'
        expectedErrorCode = errno.errorcode[socketErrorNumber]
        matchingMessages = [msg.get('log_format') == expectedFormat and msg.get('acceptError') == expectedErrorCode for msg in self.messages]
        self.assertGreater(len(matchingMessages), 0, 'Log event for failed accept not found in %r' % (self.messages,))

    def test_tooManyFilesFromAccept(self):
        if False:
            return 10
        "\n        C{accept(2)} can fail with C{EMFILE} when there are too many open file\n        descriptors in the process.  Test that this doesn't negatively impact\n        any other existing connections.\n\n        C{EMFILE} mainly occurs on Linux when the open file rlimit is\n        encountered.\n        "
        return self._acceptFailureTest(EMFILE)

    def test_noBufferSpaceFromAccept(self):
        if False:
            print('Hello World!')
        '\n        Similar to L{test_tooManyFilesFromAccept}, but test the case where\n        C{accept(2)} fails with C{ENOBUFS}.\n\n        This mainly occurs on Windows and FreeBSD, but may be possible on\n        Linux and other platforms as well.\n        '
        return self._acceptFailureTest(ENOBUFS)

    def test_connectionAbortedFromAccept(self):
        if False:
            print('Hello World!')
        '\n        Similar to L{test_tooManyFilesFromAccept}, but test the case where\n        C{accept(2)} fails with C{ECONNABORTED}.\n\n        It is not clear whether this is actually possible for TCP\n        connections on modern versions of Linux.\n        '
        return self._acceptFailureTest(ECONNABORTED)

    @skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate ENFILE')
    def test_noFilesFromAccept(self):
        if False:
            return 10
        '\n        Similar to L{test_tooManyFilesFromAccept}, but test the case where\n        C{accept(2)} fails with C{ENFILE}.\n\n        This can occur on Linux when the system has exhausted (!) its supply\n        of inodes.\n        '
        return self._acceptFailureTest(ENFILE)

    @skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate ENOMEM')
    def test_noMemoryFromAccept(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Similar to L{test_tooManyFilesFromAccept}, but test the case where\n        C{accept(2)} fails with C{ENOMEM}.\n\n        On Linux at least, this can sensibly occur, even in a Python program\n        (which eats memory like no ones business), when memory has become\n        fragmented or low memory has been filled (d_alloc calls\n        kmem_cache_alloc calls kmalloc - kmalloc only allocates out of low\n        memory).\n        '
        return self._acceptFailureTest(ENOMEM)

    @skipIf(os.environ.get('INFRASTRUCTURE') == 'AZUREPIPELINES', 'Hangs on Azure Pipelines due to firewall')
    def test_acceptScaling(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{tcp.Port.doRead} increases the number of consecutive\n        C{accept} calls it performs if all of the previous C{accept}\n        calls succeed; otherwise, it reduces the number to the amount\n        of successful calls.\n        '
        factory = ServerFactory()
        factory.protocol = Protocol
        port = self.port(0, factory, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        clients = []

        def closeAll():
            if False:
                for i in range(10):
                    print('nop')
            for client in clients:
                client.close()
        self.addCleanup(closeAll)

        def connect():
            if False:
                print('Hello World!')
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(('127.0.0.1', port.getHost().port))
            return client
        clients.append(connect())
        port.numberAccepts = 1
        port.doRead()
        self.assertGreater(port.numberAccepts, 1)
        clients.append(connect())
        port.doRead()
        self.assertEqual(port.numberAccepts, 1)
        port.doRead()
        self.assertEqual(port.numberAccepts, 1)

    @skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate EPERM')
    def test_permissionFailure(self):
        if False:
            return 10
        '\n        C{accept(2)} returning C{EPERM} is treated as a transient\n        failure and the call retried no more than the maximum number\n        of consecutive C{accept(2)} calls.\n        '
        maximumNumberOfAccepts = 123
        acceptCalls = [0]

        class FakeSocketWithAcceptLimit:
            """
            Pretend to be a socket in an overloaded system whose
            C{accept} method can only be called
            C{maximumNumberOfAccepts} times.
            """

            def accept(oself):
                if False:
                    while True:
                        i = 10
                acceptCalls[0] += 1
                if acceptCalls[0] > maximumNumberOfAccepts:
                    self.fail('Maximum number of accept calls exceeded.')
                raise OSError(EPERM, os.strerror(EPERM))
        for _ in range(maximumNumberOfAccepts):
            self.assertRaises(socket.error, FakeSocketWithAcceptLimit().accept)
        self.assertRaises(self.failureException, FakeSocketWithAcceptLimit().accept)
        acceptCalls = [0]
        factory = ServerFactory()
        port = self.port(0, factory, interface='127.0.0.1')
        port.numberAccepts = 123
        self.patch(port, 'socket', FakeSocketWithAcceptLimit())
        port.doRead()
        self.assertEquals(port.numberAccepts, 1)

    def test_unknownSocketErrorRaise(self):
        if False:
            while True:
                i = 10
        '\n        A C{socket.error} raised by C{accept(2)} whose C{errno} is\n        unknown to the recovery logic is logged.\n        '
        knownErrors = list(_ACCEPT_ERRORS)
        knownErrors.extend([EAGAIN, EPERM, EWOULDBLOCK])
        unknownAcceptError = max((error for error in knownErrors if isinstance(error, int))) + 1

        class FakeSocketWithUnknownAcceptError:
            """
            Pretend to be a socket in an overloaded system whose
            C{accept} method can only be called
            C{maximumNumberOfAccepts} times.
            """

            def accept(oself):
                if False:
                    i = 10
                    return i + 15
                raise OSError(unknownAcceptError, 'unknown socket error message')
        factory = ServerFactory()
        port = self.port(0, factory, interface='127.0.0.1')
        self.patch(port, 'socket', FakeSocketWithUnknownAcceptError())
        port.doRead()
        failures = self.flushLoggedErrors(socket.error)
        self.assertEqual(1, len(failures))
        self.assertEqual(failures[0].value.args[0], unknownAcceptError)