"""
An epoll() based implementation of the twisted main loop.

To install the event loop (and you should do this before any connections,
listeners or connectors are added)::

    from twisted.internet import epollreactor
    epollreactor.install()
"""
import errno
import select
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
try:
    epoll = getattr(select, 'epoll')
    EPOLLHUP = getattr(select, 'EPOLLHUP')
    EPOLLERR = getattr(select, 'EPOLLERR')
    EPOLLIN = getattr(select, 'EPOLLIN')
    EPOLLOUT = getattr(select, 'EPOLLOUT')
except AttributeError as e:
    raise ImportError(e)

@implementer(IReactorFDSet)
class EPollReactor(posixbase.PosixReactorBase, posixbase._PollLikeMixin):
    """
    A reactor that uses epoll(7).

    @ivar _poller: A C{epoll} which will be used to check for I/O
        readiness.

    @ivar _selectables: A dictionary mapping integer file descriptors to
        instances of C{FileDescriptor} which have been registered with the
        reactor.  All C{FileDescriptors} which are currently receiving read or
        write readiness notifications will be present as values in this
        dictionary.

    @ivar _reads: A set containing integer file descriptors.  Values in this
        set will be registered with C{_poller} for read readiness notifications
        which will be dispatched to the corresponding C{FileDescriptor}
        instances in C{_selectables}.

    @ivar _writes: A set containing integer file descriptors.  Values in this
        set will be registered with C{_poller} for write readiness
        notifications which will be dispatched to the corresponding
        C{FileDescriptor} instances in C{_selectables}.

    @ivar _continuousPolling: A L{_ContinuousPolling} instance, used to handle
        file descriptors (e.g. filesystem files) that are not supported by
        C{epoll(7)}.
    """
    _POLL_DISCONNECTED = EPOLLHUP | EPOLLERR
    _POLL_IN = EPOLLIN
    _POLL_OUT = EPOLLOUT

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize epoll object, file descriptor tracking dictionaries, and the\n        base class.\n        '
        self._poller = epoll(1024)
        self._reads = set()
        self._writes = set()
        self._selectables = {}
        self._continuousPolling = posixbase._ContinuousPolling(self)
        posixbase.PosixReactorBase.__init__(self)

    def _add(self, xer, primary, other, selectables, event, antievent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Private method for adding a descriptor from the event loop.\n\n        It takes care of adding it if  new or modifying it if already added\n        for another state (read -> read/write for example).\n        '
        fd = xer.fileno()
        if fd not in primary:
            flags = event
            if fd in other:
                flags |= antievent
                self._poller.modify(fd, flags)
            else:
                self._poller.register(fd, flags)
            primary.add(fd)
            selectables[fd] = xer

    def addReader(self, reader):
        if False:
            print('Hello World!')
        '\n        Add a FileDescriptor for notification of data available to read.\n        '
        try:
            self._add(reader, self._reads, self._writes, self._selectables, EPOLLIN, EPOLLOUT)
        except OSError as e:
            if e.errno == errno.EPERM:
                self._continuousPolling.addReader(reader)
            else:
                raise

    def addWriter(self, writer):
        if False:
            return 10
        '\n        Add a FileDescriptor for notification of data available to write.\n        '
        try:
            self._add(writer, self._writes, self._reads, self._selectables, EPOLLOUT, EPOLLIN)
        except OSError as e:
            if e.errno == errno.EPERM:
                self._continuousPolling.addWriter(writer)
            else:
                raise

    def _remove(self, xer, primary, other, selectables, event, antievent):
        if False:
            print('Hello World!')
        '\n        Private method for removing a descriptor from the event loop.\n\n        It does the inverse job of _add, and also add a check in case of the fd\n        has gone away.\n        '
        fd = xer.fileno()
        if fd == -1:
            for (fd, fdes) in selectables.items():
                if xer is fdes:
                    break
            else:
                return
        if fd in primary:
            if fd in other:
                flags = antievent
                self._poller.modify(fd, flags)
            else:
                del selectables[fd]
                self._poller.unregister(fd)
            primary.remove(fd)

    def removeReader(self, reader):
        if False:
            print('Hello World!')
        '\n        Remove a Selectable for notification of data available to read.\n        '
        if self._continuousPolling.isReading(reader):
            self._continuousPolling.removeReader(reader)
            return
        self._remove(reader, self._reads, self._writes, self._selectables, EPOLLIN, EPOLLOUT)

    def removeWriter(self, writer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove a Selectable for notification of data available to write.\n        '
        if self._continuousPolling.isWriting(writer):
            self._continuousPolling.removeWriter(writer)
            return
        self._remove(writer, self._writes, self._reads, self._selectables, EPOLLOUT, EPOLLIN)

    def removeAll(self):
        if False:
            i = 10
            return i + 15
        '\n        Remove all selectables, and return a list of them.\n        '
        return self._removeAll([self._selectables[fd] for fd in self._reads], [self._selectables[fd] for fd in self._writes]) + self._continuousPolling.removeAll()

    def getReaders(self):
        if False:
            print('Hello World!')
        return [self._selectables[fd] for fd in self._reads] + self._continuousPolling.getReaders()

    def getWriters(self):
        if False:
            return 10
        return [self._selectables[fd] for fd in self._writes] + self._continuousPolling.getWriters()

    def doPoll(self, timeout):
        if False:
            return 10
        '\n        Poll the poller for new events.\n        '
        if timeout is None:
            timeout = -1
        try:
            l = self._poller.poll(timeout, len(self._selectables))
        except OSError as err:
            if err.errno == errno.EINTR:
                return
            raise
        _drdw = self._doReadOrWrite
        for (fd, event) in l:
            try:
                selectable = self._selectables[fd]
            except KeyError:
                pass
            else:
                log.callWithLogger(selectable, _drdw, selectable, fd, event)
    doIteration = doPoll

def install():
    if False:
        i = 10
        return i + 15
    '\n    Install the epoll() reactor.\n    '
    p = EPollReactor()
    from twisted.internet.main import installReactor
    installReactor(p)
__all__ = ['EPollReactor', 'install']