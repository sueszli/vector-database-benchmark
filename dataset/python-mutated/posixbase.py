"""
Posix reactor base class
"""
import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import IHalfCloseableDescriptor, IReactorFDSet, IReactorMulticast, IReactorProcess, IReactorSocket, IReactorSSL, IReactorTCP, IReactorUDP, IReactorUNIX, IReactorUNIXDatagram
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import SignalHandling, _ChildSignalHandling, _IWaker, _MultiSignalHandling, _Waker
_NO_FILENO = error.ConnectionFdescWentAway('Handler has no fileno method')
_NO_FILEDESC = error.ConnectionFdescWentAway('File descriptor lost')
try:
    from twisted.protocols import tls as _tls
except ImportError:
    tls = None
else:
    tls = _tls
try:
    from twisted.internet import ssl as _ssl
except ImportError:
    ssl = None
else:
    ssl = _ssl
unixEnabled = platformType == 'posix'
processEnabled = False
if unixEnabled:
    from twisted.internet import process, unix
    processEnabled = True
if platform.isWindows():
    try:
        import win32process
        processEnabled = True
    except ImportError:
        win32process = None

class _DisconnectSelectableMixin:
    """
    Mixin providing the C{_disconnectSelectable} method.
    """

    def _disconnectSelectable(self, selectable, why, isRead, faildict={error.ConnectionDone: failure.Failure(error.ConnectionDone()), error.ConnectionLost: failure.Failure(error.ConnectionLost())}):
        if False:
            i = 10
            return i + 15
        '\n        Utility function for disconnecting a selectable.\n\n        Supports half-close notification, isRead should be boolean indicating\n        whether error resulted from doRead().\n        '
        self.removeReader(selectable)
        f = faildict.get(why.__class__)
        if f:
            if isRead and why.__class__ == error.ConnectionDone and IHalfCloseableDescriptor.providedBy(selectable):
                selectable.readConnectionLost(f)
            else:
                self.removeWriter(selectable)
                selectable.connectionLost(f)
        else:
            self.removeWriter(selectable)
            selectable.connectionLost(failure.Failure(why))

@implementer(IReactorTCP, IReactorUDP, IReactorMulticast)
class PosixReactorBase(_DisconnectSelectableMixin, ReactorBase):
    """
    A basis for reactors that use file descriptors.

    @ivar _childWaker: L{None} or a reference to the L{_SIGCHLDWaker}
        which is used to properly notice child process termination.
    """
    _childWaker = None

    def _wakerFactory(self) -> _IWaker:
        if False:
            return 10
        return _Waker()

    def installWaker(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Install a `waker' to allow threads and signals to wake up the IO thread.\n\n        We use the self-pipe trick (http://cr.yp.to/docs/selfpipe.html) to wake\n        the reactor. On Windows we use a pair of sockets.\n        "
        if not self.waker:
            self.waker = self._wakerFactory()
            self._internalReaders.add(self.waker)
            self.addReader(self.waker)

    def _signalsFactory(self) -> SignalHandling:
        if False:
            print('Hello World!')
        '\n        Customize reactor signal handling to support child processes on POSIX\n        platforms.\n        '
        baseHandling = super()._signalsFactory()
        if platformType == 'posix':
            return _MultiSignalHandling((baseHandling, _ChildSignalHandling(self._addInternalReader, self._removeInternalReader)))
        return baseHandling

    def spawnProcess(self, processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        if False:
            return 10
        if platformType == 'posix':
            if usePTY:
                if childFDs is not None:
                    raise ValueError('Using childFDs is not supported with usePTY=True.')
                return process.PTYProcess(self, executable, args, env, path, processProtocol, uid, gid, usePTY)
            else:
                return process.Process(self, executable, args, env, path, processProtocol, uid, gid, childFDs)
        elif platformType == 'win32':
            if uid is not None:
                raise ValueError('Setting UID is unsupported on this platform.')
            if gid is not None:
                raise ValueError('Setting GID is unsupported on this platform.')
            if usePTY:
                raise ValueError('The usePTY parameter is not supported on Windows.')
            if childFDs:
                raise ValueError('Customizing childFDs is not supported on Windows.')
            if win32process:
                from twisted.internet._dumbwin32proc import Process
                return Process(self, processProtocol, executable, args, env, path)
            else:
                raise NotImplementedError('spawnProcess not available since pywin32 is not installed.')
        else:
            raise NotImplementedError('spawnProcess only available on Windows or POSIX.')

    def listenUDP(self, port, protocol, interface='', maxPacketSize=8192):
        if False:
            for i in range(10):
                print('nop')
        'Connects a given L{DatagramProtocol} to the given numeric UDP port.\n\n        @returns: object conforming to L{IListeningPort}.\n        '
        p = udp.Port(port, protocol, interface, maxPacketSize, self)
        p.startListening()
        return p

    def listenMulticast(self, port, protocol, interface='', maxPacketSize=8192, listenMultiple=False):
        if False:
            i = 10
            return i + 15
        'Connects a given DatagramProtocol to the given numeric UDP port.\n\n        EXPERIMENTAL.\n\n        @returns: object conforming to IListeningPort.\n        '
        p = udp.MulticastPort(port, protocol, interface, maxPacketSize, self, listenMultiple)
        p.startListening()
        return p

    def connectUNIX(self, address, factory, timeout=30, checkPID=0):
        if False:
            return 10
        assert unixEnabled, 'UNIX support is not present'
        c = unix.Connector(address, factory, timeout, self, checkPID)
        c.connect()
        return c

    def listenUNIX(self, address, factory, backlog=50, mode=438, wantPID=0):
        if False:
            for i in range(10):
                print('nop')
        assert unixEnabled, 'UNIX support is not present'
        p = unix.Port(address, factory, backlog, mode, self, wantPID)
        p.startListening()
        return p

    def listenUNIXDatagram(self, address, protocol, maxPacketSize=8192, mode=438):
        if False:
            i = 10
            return i + 15
        '\n        Connects a given L{DatagramProtocol} to the given path.\n\n        EXPERIMENTAL.\n\n        @returns: object conforming to L{IListeningPort}.\n        '
        assert unixEnabled, 'UNIX support is not present'
        p = unix.DatagramPort(address, protocol, maxPacketSize, mode, self)
        p.startListening()
        return p

    def connectUNIXDatagram(self, address, protocol, maxPacketSize=8192, mode=438, bindAddress=None):
        if False:
            i = 10
            return i + 15
        '\n        Connects a L{ConnectedDatagramProtocol} instance to a path.\n\n        EXPERIMENTAL.\n        '
        assert unixEnabled, 'UNIX support is not present'
        p = unix.ConnectedDatagramPort(address, protocol, maxPacketSize, mode, bindAddress, self)
        p.startListening()
        return p
    if unixEnabled:
        _supportedAddressFamilies: Sequence[socket.AddressFamily] = (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX)
    else:
        _supportedAddressFamilies = (socket.AF_INET, socket.AF_INET6)

    def adoptStreamPort(self, fileDescriptor, addressFamily, factory):
        if False:
            return 10
        '\n        Create a new L{IListeningPort} from an already-initialized socket.\n\n        This just dispatches to a suitable port implementation (eg from\n        L{IReactorTCP}, etc) based on the specified C{addressFamily}.\n\n        @see: L{twisted.internet.interfaces.IReactorSocket.adoptStreamPort}\n        '
        if addressFamily not in self._supportedAddressFamilies:
            raise error.UnsupportedAddressFamily(addressFamily)
        if unixEnabled and addressFamily == socket.AF_UNIX:
            p = unix.Port._fromListeningDescriptor(self, fileDescriptor, factory)
        else:
            p = tcp.Port._fromListeningDescriptor(self, fileDescriptor, addressFamily, factory)
        p.startListening()
        return p

    def adoptStreamConnection(self, fileDescriptor, addressFamily, factory):
        if False:
            while True:
                i = 10
        '\n        @see:\n            L{twisted.internet.interfaces.IReactorSocket.adoptStreamConnection}\n        '
        if addressFamily not in self._supportedAddressFamilies:
            raise error.UnsupportedAddressFamily(addressFamily)
        if unixEnabled and addressFamily == socket.AF_UNIX:
            return unix.Server._fromConnectedSocket(fileDescriptor, factory, self)
        else:
            return tcp.Server._fromConnectedSocket(fileDescriptor, addressFamily, factory, self)

    def adoptDatagramPort(self, fileDescriptor, addressFamily, protocol, maxPacketSize=8192):
        if False:
            i = 10
            return i + 15
        if addressFamily not in (socket.AF_INET, socket.AF_INET6):
            raise error.UnsupportedAddressFamily(addressFamily)
        p = udp.Port._fromListeningDescriptor(self, fileDescriptor, addressFamily, protocol, maxPacketSize=maxPacketSize)
        p.startListening()
        return p

    def listenTCP(self, port, factory, backlog=50, interface=''):
        if False:
            while True:
                i = 10
        p = tcp.Port(port, factory, backlog, interface, self)
        p.startListening()
        return p

    def connectTCP(self, host, port, factory, timeout=30, bindAddress=None):
        if False:
            i = 10
            return i + 15
        c = tcp.Connector(host, port, factory, timeout, bindAddress, self)
        c.connect()
        return c

    def connectSSL(self, host, port, factory, contextFactory, timeout=30, bindAddress=None):
        if False:
            for i in range(10):
                print('nop')
        if tls is not None:
            tlsFactory = tls.TLSMemoryBIOFactory(contextFactory, True, factory)
            return self.connectTCP(host, port, tlsFactory, timeout, bindAddress)
        elif ssl is not None:
            c = ssl.Connector(host, port, factory, contextFactory, timeout, bindAddress, self)
            c.connect()
            return c
        else:
            assert False, 'SSL support is not present'

    def listenSSL(self, port, factory, contextFactory, backlog=50, interface=''):
        if False:
            i = 10
            return i + 15
        if tls is not None:
            tlsFactory = tls.TLSMemoryBIOFactory(contextFactory, False, factory)
            port = self.listenTCP(port, tlsFactory, backlog, interface)
            port._type = 'TLS'
            return port
        elif ssl is not None:
            p = ssl.Port(port, factory, contextFactory, backlog, interface, self)
            p.startListening()
            return p
        else:
            assert False, 'SSL support is not present'

    def _removeAll(self, readers, writers):
        if False:
            print('Hello World!')
        '\n        Remove all readers and writers, and list of removed L{IReadDescriptor}s\n        and L{IWriteDescriptor}s.\n\n        Meant for calling from subclasses, to implement removeAll, like::\n\n          def removeAll(self):\n              return self._removeAll(self._reads, self._writes)\n\n        where C{self._reads} and C{self._writes} are iterables.\n        '
        removedReaders = set(readers) - self._internalReaders
        for reader in removedReaders:
            self.removeReader(reader)
        removedWriters = set(writers)
        for writer in removedWriters:
            self.removeWriter(writer)
        return list(removedReaders | removedWriters)

class _PollLikeMixin:
    """
    Mixin for poll-like reactors.

    Subclasses must define the following attributes::

      - _POLL_DISCONNECTED - Bitmask for events indicating a connection was
        lost.
      - _POLL_IN - Bitmask for events indicating there is input to read.
      - _POLL_OUT - Bitmask for events indicating output can be written.

    Must be mixed in to a subclass of PosixReactorBase (for
    _disconnectSelectable).
    """

    def _doReadOrWrite(self, selectable, fd, event):
        if False:
            return 10
        '\n        fd is available for read or write, do the work and raise errors if\n        necessary.\n        '
        why = None
        inRead = False
        if event & self._POLL_DISCONNECTED and (not event & self._POLL_IN):
            if fd in self._reads:
                inRead = True
                why = CONNECTION_DONE
            else:
                why = CONNECTION_LOST
        else:
            try:
                if selectable.fileno() == -1:
                    why = _NO_FILEDESC
                else:
                    if event & self._POLL_IN:
                        why = selectable.doRead()
                        inRead = True
                    if not why and event & self._POLL_OUT:
                        why = selectable.doWrite()
                        inRead = False
            except BaseException:
                why = sys.exc_info()[1]
                log.err()
        if why:
            self._disconnectSelectable(selectable, why, inRead)

@implementer(IReactorFDSet)
class _ContinuousPolling(_PollLikeMixin, _DisconnectSelectableMixin):
    """
    Schedule reads and writes based on the passage of time, rather than
    notification.

    This is useful for supporting polling filesystem files, which C{epoll(7)}
    does not support.

    The implementation uses L{_PollLikeMixin}, which is a bit hacky, but
    re-implementing and testing the relevant code yet again is unappealing.

    @ivar _reactor: The L{EPollReactor} that is using this instance.

    @ivar _loop: A C{LoopingCall} that drives the polling, or L{None}.

    @ivar _readers: A C{set} of C{FileDescriptor} objects that should be read
        from.

    @ivar _writers: A C{set} of C{FileDescriptor} objects that should be
        written to.
    """
    _POLL_DISCONNECTED = 1
    _POLL_IN = 2
    _POLL_OUT = 4

    def __init__(self, reactor):
        if False:
            for i in range(10):
                print('nop')
        self._reactor = reactor
        self._loop = None
        self._readers = set()
        self._writers = set()

    def _checkLoop(self):
        if False:
            return 10
        '\n        Start or stop a C{LoopingCall} based on whether there are readers and\n        writers.\n        '
        if self._readers or self._writers:
            if self._loop is None:
                from twisted.internet.task import _EPSILON, LoopingCall
                self._loop = LoopingCall(self.iterate)
                self._loop.clock = self._reactor
                self._loop.start(_EPSILON, now=False)
        elif self._loop:
            self._loop.stop()
            self._loop = None

    def iterate(self):
        if False:
            i = 10
            return i + 15
        '\n        Call C{doRead} and C{doWrite} on all readers and writers respectively.\n        '
        for reader in list(self._readers):
            self._doReadOrWrite(reader, reader, self._POLL_IN)
        for writer in list(self._writers):
            self._doReadOrWrite(writer, writer, self._POLL_OUT)

    def addReader(self, reader):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a C{FileDescriptor} for notification of data available to read.\n        '
        self._readers.add(reader)
        self._checkLoop()

    def addWriter(self, writer):
        if False:
            print('Hello World!')
        '\n        Add a C{FileDescriptor} for notification of data available to write.\n        '
        self._writers.add(writer)
        self._checkLoop()

    def removeReader(self, reader):
        if False:
            i = 10
            return i + 15
        '\n        Remove a C{FileDescriptor} from notification of data available to read.\n        '
        try:
            self._readers.remove(reader)
        except KeyError:
            return
        self._checkLoop()

    def removeWriter(self, writer):
        if False:
            print('Hello World!')
        '\n        Remove a C{FileDescriptor} from notification of data available to\n        write.\n        '
        try:
            self._writers.remove(writer)
        except KeyError:
            return
        self._checkLoop()

    def removeAll(self):
        if False:
            while True:
                i = 10
        '\n        Remove all readers and writers.\n        '
        result = list(self._readers | self._writers)
        self._readers.clear()
        self._writers.clear()
        return result

    def getReaders(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of the readers.\n        '
        return list(self._readers)

    def getWriters(self):
        if False:
            print('Hello World!')
        '\n        Return a list of the writers.\n        '
        return list(self._writers)

    def isReading(self, fd):
        if False:
            while True:
                i = 10
        '\n        Checks if the file descriptor is currently being observed for read\n        readiness.\n\n        @param fd: The file descriptor being checked.\n        @type fd: L{twisted.internet.abstract.FileDescriptor}\n        @return: C{True} if the file descriptor is being observed for read\n            readiness, C{False} otherwise.\n        @rtype: C{bool}\n        '
        return fd in self._readers

    def isWriting(self, fd):
        if False:
            i = 10
            return i + 15
        '\n        Checks if the file descriptor is currently being observed for write\n        readiness.\n\n        @param fd: The file descriptor being checked.\n        @type fd: L{twisted.internet.abstract.FileDescriptor}\n        @return: C{True} if the file descriptor is being observed for write\n            readiness, C{False} otherwise.\n        @rtype: C{bool}\n        '
        return fd in self._writers
if tls is not None or ssl is not None:
    classImplements(PosixReactorBase, IReactorSSL)
if unixEnabled:
    classImplements(PosixReactorBase, IReactorUNIX, IReactorUNIXDatagram)
if processEnabled:
    classImplements(PosixReactorBase, IReactorProcess)
if getattr(socket, 'fromfd', None) is not None:
    classImplements(PosixReactorBase, IReactorSocket)
__all__ = ['PosixReactorBase']