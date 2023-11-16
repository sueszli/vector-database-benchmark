"""
Tests for implementations of L{IReactorFDSet}.
"""
import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest

def socketpair():
    if False:
        print('Hello World!')
    serverSocket = socket.socket()
    serverSocket.bind(('127.0.0.1', 0))
    serverSocket.listen(1)
    try:
        client = socket.socket()
        try:
            client.setblocking(False)
            try:
                client.connect(('127.0.0.1', serverSocket.getsockname()[1]))
            except OSError as e:
                if e.args[0] not in (EINPROGRESS, EWOULDBLOCK):
                    raise
            (server, addr) = serverSocket.accept()
        except BaseException:
            client.close()
            raise
    finally:
        serverSocket.close()
    return (client, server)

class ReactorFDSetTestsBuilder(ReactorBuilder):
    """
    Builder defining tests relating to L{IReactorFDSet}.
    """
    requiredInterfaces = [IReactorFDSet]

    def _connectedPair(self):
        if False:
            print('Hello World!')
        '\n        Return the two sockets which make up a new TCP connection.\n        '
        (client, server) = socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        return (client, server)

    def _simpleSetup(self):
        if False:
            i = 10
            return i + 15
        reactor = self.buildReactor()
        (client, server) = self._connectedPair()
        fd = FileDescriptor(reactor)
        fd.fileno = client.fileno
        return (reactor, fd, server)

    def test_addReader(self):
        if False:
            while True:
                i = 10
        '\n        C{reactor.addReader()} accepts an L{IReadDescriptor} provider and calls\n        its C{doRead} method when there may be data available on its C{fileno}.\n        '
        (reactor, fd, server) = self._simpleSetup()

        def removeAndStop():
            if False:
                return 10
            reactor.removeReader(fd)
            reactor.stop()
        fd.doRead = removeAndStop
        reactor.addReader(fd)
        server.sendall(b'x')
        self.runReactor(reactor)

    def test_removeReader(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{reactor.removeReader()} accepts an L{IReadDescriptor} provider\n        previously passed to C{reactor.addReader()} and causes it to no longer\n        be monitored for input events.\n        '
        (reactor, fd, server) = self._simpleSetup()

        def fail():
            if False:
                return 10
            self.fail('doRead should not be called')
        fd.doRead = fail
        reactor.addReader(fd)
        reactor.removeReader(fd)
        server.sendall(b'x')
        reactor.callLater(0, reactor.callLater, 0, reactor.stop)
        self.runReactor(reactor)

    def test_addWriter(self):
        if False:
            while True:
                i = 10
        '\n        C{reactor.addWriter()} accepts an L{IWriteDescriptor} provider and\n        calls its C{doWrite} method when it may be possible to write to its\n        C{fileno}.\n        '
        (reactor, fd, server) = self._simpleSetup()

        def removeAndStop():
            if False:
                while True:
                    i = 10
            reactor.removeWriter(fd)
            reactor.stop()
        fd.doWrite = removeAndStop
        reactor.addWriter(fd)
        self.runReactor(reactor)

    def _getFDTest(self, kind):
        if False:
            i = 10
            return i + 15
        '\n        Helper for getReaders and getWriters tests.\n        '
        reactor = self.buildReactor()
        get = getattr(reactor, 'get' + kind + 's')
        add = getattr(reactor, 'add' + kind)
        remove = getattr(reactor, 'remove' + kind)
        (client, server) = self._connectedPair()
        self.assertNotIn(client, get())
        self.assertNotIn(server, get())
        add(client)
        self.assertIn(client, get())
        self.assertNotIn(server, get())
        remove(client)
        self.assertNotIn(client, get())
        self.assertNotIn(server, get())

    def test_getReaders(self):
        if False:
            print('Hello World!')
        '\n        L{IReactorFDSet.getReaders} reflects the additions and removals made\n        with L{IReactorFDSet.addReader} and L{IReactorFDSet.removeReader}.\n        '
        self._getFDTest('Reader')

    def test_removeWriter(self):
        if False:
            return 10
        '\n        L{reactor.removeWriter()} accepts an L{IWriteDescriptor} provider\n        previously passed to C{reactor.addWriter()} and causes it to no longer\n        be monitored for outputability.\n        '
        (reactor, fd, server) = self._simpleSetup()

        def fail():
            if False:
                i = 10
                return i + 15
            self.fail('doWrite should not be called')
        fd.doWrite = fail
        reactor.addWriter(fd)
        reactor.removeWriter(fd)
        reactor.callLater(0, reactor.callLater, 0, reactor.stop)
        self.runReactor(reactor)

    def test_getWriters(self):
        if False:
            return 10
        '\n        L{IReactorFDSet.getWriters} reflects the additions and removals made\n        with L{IReactorFDSet.addWriter} and L{IReactorFDSet.removeWriter}.\n        '
        self._getFDTest('Writer')

    def test_removeAll(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        C{reactor.removeAll()} removes all registered L{IReadDescriptor}\n        providers and all registered L{IWriteDescriptor} providers and returns\n        them.\n        '
        reactor = self.buildReactor()
        (reactor, fd, server) = self._simpleSetup()
        fd.doRead = lambda : self.fail('doRead should not be called')
        fd.doWrite = lambda : self.fail('doWrite should not be called')
        server.sendall(b'x')
        reactor.addReader(fd)
        reactor.addWriter(fd)
        removed = reactor.removeAll()
        reactor.callLater(0, reactor.callLater, 0, reactor.stop)
        self.runReactor(reactor)
        self.assertEqual(removed, [fd])

    def test_removedFromReactor(self):
        if False:
            i = 10
            return i + 15
        "\n        A descriptor's C{fileno} method should not be called after the\n        descriptor has been removed from the reactor.\n        "
        reactor = self.buildReactor()
        descriptor = RemovingDescriptor(reactor)
        reactor.callWhenRunning(descriptor.start)
        self.runReactor(reactor)
        self.assertEqual(descriptor.calls, [])

    def test_negativeOneFileDescriptor(self):
        if False:
            i = 10
            return i + 15
        '\n        If L{FileDescriptor.fileno} returns C{-1}, the descriptor is removed\n        from the reactor.\n        '
        reactor = self.buildReactor()
        (client, server) = self._connectedPair()

        class DisappearingDescriptor(FileDescriptor):
            _fileno = server.fileno()
            _received = b''

            def fileno(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._fileno

            def doRead(self):
                if False:
                    while True:
                        i = 10
                self._fileno = -1
                self._received += server.recv(1)
                client.send(b'y')

            def connectionLost(self, reason):
                if False:
                    print('Hello World!')
                reactor.stop()
        descriptor = DisappearingDescriptor(reactor)
        reactor.addReader(descriptor)
        client.send(b'x')
        self.runReactor(reactor)
        self.assertEqual(descriptor._received, b'x')

    @skipIf(platform.isWindows(), 'Cannot duplicate socket filenos on Windows')
    def test_lostFileDescriptor(self):
        if False:
            return 10
        '\n        The file descriptor underlying a FileDescriptor may be closed and\n        replaced by another at some point.  Bytes which arrive on the new\n        descriptor must not be delivered to the FileDescriptor which was\n        originally registered with the original descriptor of the same number.\n\n        Practically speaking, this is difficult or impossible to detect.  The\n        implementation relies on C{fileno} raising an exception if the original\n        descriptor has gone away.  If C{fileno} continues to return the original\n        file descriptor value, the reactor may deliver events from that\n        descriptor.  This is a best effort attempt to ease certain debugging\n        situations.  Applications should not rely on it intentionally.\n        '
        reactor = self.buildReactor()
        name = reactor.__class__.__name__
        if name in ('EPollReactor', 'KQueueReactor', 'CFReactor', 'AsyncioSelectorReactor'):
            raise SkipTest(f'{name!r} cannot detect lost file descriptors')
        (client, server) = self._connectedPair()

        class Victim(FileDescriptor):
            """
            This L{FileDescriptor} will have its socket closed out from under it
            and another socket will take its place.  It will raise a
            socket.error from C{fileno} after this happens (because socket
            objects remember whether they have been closed), so as long as the
            reactor calls the C{fileno} method the problem will be detected.
            """

            def fileno(self):
                if False:
                    return 10
                return server.fileno()

            def doRead(self):
                if False:
                    while True:
                        i = 10
                raise Exception('Victim.doRead should never be called')

            def connectionLost(self, reason):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                When the problem is detected, the reactor should disconnect this\n                file descriptor.  When that happens, stop the reactor so the\n                test ends.\n                '
                reactor.stop()
        reactor.addReader(Victim())

        def messItUp():
            if False:
                for i in range(10):
                    print('nop')
            (newC, newS) = self._connectedPair()
            fileno = server.fileno()
            server.close()
            os.dup2(newS.fileno(), fileno)
            newC.send(b'x')
        reactor.callLater(0, messItUp)
        self.runReactor(reactor)
        self.flushLoggedErrors(socket.error)

    def test_connectionLostOnShutdown(self):
        if False:
            print('Hello World!')
        '\n        Any file descriptors added to the reactor have their C{connectionLost}\n        called when C{reactor.stop} is called.\n        '
        reactor = self.buildReactor()

        class DoNothingDescriptor(FileDescriptor):

            def doRead(self):
                if False:
                    print('Hello World!')
                return None

            def doWrite(self):
                if False:
                    while True:
                        i = 10
                return None
        (client, server) = self._connectedPair()
        fd1 = DoNothingDescriptor(reactor)
        fd1.fileno = client.fileno
        fd2 = DoNothingDescriptor(reactor)
        fd2.fileno = server.fileno
        reactor.addReader(fd1)
        reactor.addWriter(fd2)
        reactor.callWhenRunning(reactor.stop)
        self.runReactor(reactor)
        self.assertTrue(fd1.disconnected)
        self.assertTrue(fd2.disconnected)

@implementer(IReadDescriptor)
class RemovingDescriptor:
    """
    A read descriptor which removes itself from the reactor as soon as it
    gets a chance to do a read and keeps track of when its own C{fileno}
    method is called.

    @ivar insideReactor: A flag which is true as long as the reactor has
        this descriptor as a reader.

    @ivar calls: A list of the bottom of the call stack for any call to
        C{fileno} when C{insideReactor} is false.
    """

    def __init__(self, reactor):
        if False:
            for i in range(10):
                print('nop')
        self.reactor = reactor
        self.insideReactor = False
        self.calls = []
        (self.read, self.write) = socketpair()

    def start(self):
        if False:
            i = 10
            return i + 15
        self.insideReactor = True
        self.reactor.addReader(self)
        self.write.send(b'a')

    def logPrefix(self):
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    def doRead(self):
        if False:
            print('Hello World!')
        self.reactor.removeReader(self)
        self.insideReactor = False
        self.reactor.stop()
        self.read.close()
        self.write.close()

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.insideReactor:
            self.calls.append(traceback.extract_stack(limit=5)[:-1])
        return self.read.fileno()

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        pass
globals().update(ReactorFDSetTestsBuilder.makeTestCaseClasses())