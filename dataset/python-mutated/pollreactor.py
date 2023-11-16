"""
A poll() based implementation of the twisted main loop.

To install the event loop (and you should do this before any connections,
listeners or connectors are added)::

    from twisted.internet import pollreactor
    pollreactor.install()
"""
import errno
from select import POLLERR, POLLHUP, POLLIN, POLLNVAL, POLLOUT, error as SelectError, poll
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log

@implementer(IReactorFDSet)
class PollReactor(posixbase.PosixReactorBase, posixbase._PollLikeMixin):
    """
    A reactor that uses poll(2).

    @ivar _poller: A L{select.poll} which will be used to check for I/O
        readiness.

    @ivar _selectables: A dictionary mapping integer file descriptors to
        instances of L{FileDescriptor} which have been registered with the
        reactor.  All L{FileDescriptor}s which are currently receiving read or
        write readiness notifications will be present as values in this
        dictionary.

    @ivar _reads: A dictionary mapping integer file descriptors to arbitrary
        values (this is essentially a set).  Keys in this dictionary will be
        registered with C{_poller} for read readiness notifications which will
        be dispatched to the corresponding L{FileDescriptor} instances in
        C{_selectables}.

    @ivar _writes: A dictionary mapping integer file descriptors to arbitrary
        values (this is essentially a set).  Keys in this dictionary will be
        registered with C{_poller} for write readiness notifications which will
        be dispatched to the corresponding L{FileDescriptor} instances in
        C{_selectables}.
    """
    _POLL_DISCONNECTED = POLLHUP | POLLERR | POLLNVAL
    _POLL_IN = POLLIN
    _POLL_OUT = POLLOUT

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize polling object, file descriptor tracking dictionaries, and\n        the base class.\n        '
        self._poller = poll()
        self._selectables = {}
        self._reads = {}
        self._writes = {}
        posixbase.PosixReactorBase.__init__(self)

    def _updateRegistration(self, fd):
        if False:
            while True:
                i = 10
        'Register/unregister an fd with the poller.'
        try:
            self._poller.unregister(fd)
        except KeyError:
            pass
        mask = 0
        if fd in self._reads:
            mask = mask | POLLIN
        if fd in self._writes:
            mask = mask | POLLOUT
        if mask != 0:
            self._poller.register(fd, mask)
        elif fd in self._selectables:
            del self._selectables[fd]

    def _dictRemove(self, selectable, mdict):
        if False:
            while True:
                i = 10
        try:
            fd = selectable.fileno()
            mdict[fd]
        except BaseException:
            for (fd, fdes) in self._selectables.items():
                if selectable is fdes:
                    break
            else:
                return
        if fd in mdict:
            del mdict[fd]
            self._updateRegistration(fd)

    def addReader(self, reader):
        if False:
            for i in range(10):
                print('nop')
        'Add a FileDescriptor for notification of data available to read.'
        fd = reader.fileno()
        if fd not in self._reads:
            self._selectables[fd] = reader
            self._reads[fd] = 1
            self._updateRegistration(fd)

    def addWriter(self, writer):
        if False:
            for i in range(10):
                print('nop')
        'Add a FileDescriptor for notification of data available to write.'
        fd = writer.fileno()
        if fd not in self._writes:
            self._selectables[fd] = writer
            self._writes[fd] = 1
            self._updateRegistration(fd)

    def removeReader(self, reader):
        if False:
            return 10
        'Remove a Selectable for notification of data available to read.'
        return self._dictRemove(reader, self._reads)

    def removeWriter(self, writer):
        if False:
            print('Hello World!')
        'Remove a Selectable for notification of data available to write.'
        return self._dictRemove(writer, self._writes)

    def removeAll(self):
        if False:
            while True:
                i = 10
        '\n        Remove all selectables, and return a list of them.\n        '
        return self._removeAll([self._selectables[fd] for fd in self._reads], [self._selectables[fd] for fd in self._writes])

    def doPoll(self, timeout):
        if False:
            return 10
        'Poll the poller for new events.'
        if timeout is not None:
            timeout = int(timeout * 1000)
        try:
            l = self._poller.poll(timeout)
        except SelectError as e:
            if e.args[0] == errno.EINTR:
                return
            else:
                raise
        _drdw = self._doReadOrWrite
        for (fd, event) in l:
            try:
                selectable = self._selectables[fd]
            except KeyError:
                continue
            log.callWithLogger(selectable, _drdw, selectable, fd, event)
    doIteration = doPoll

    def getReaders(self):
        if False:
            print('Hello World!')
        return [self._selectables[fd] for fd in self._reads]

    def getWriters(self):
        if False:
            return 10
        return [self._selectables[fd] for fd in self._writes]

def install():
    if False:
        return 10
    'Install the poll() reactor.'
    p = PollReactor()
    from twisted.internet.main import installReactor
    installReactor(p)
__all__ = ['PollReactor', 'install']