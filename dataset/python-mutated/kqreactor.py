"""
A kqueue()/kevent() based implementation of the Twisted main loop.

To use this reactor, start your application specifying the kqueue reactor::

   twistd --reactor kqueue ...

To install the event loop from code (and you should do this before any
connections, listeners or connectors are added)::

   from twisted.internet import kqreactor
   kqreactor.install()
"""
import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log
try:
    KQ_EV_ADD = getattr(select, 'KQ_EV_ADD')
    KQ_EV_DELETE = getattr(select, 'KQ_EV_DELETE')
    KQ_EV_EOF = getattr(select, 'KQ_EV_EOF')
    KQ_FILTER_READ = getattr(select, 'KQ_FILTER_READ')
    KQ_FILTER_WRITE = getattr(select, 'KQ_FILTER_WRITE')
except AttributeError as e:
    raise ImportError(e)

class _IKQueue(Interface):
    """
    An interface for KQueue implementations.
    """
    kqueue = Attribute('An implementation of kqueue(2).')
    kevent = Attribute('An implementation of kevent(2).')
declarations.directlyProvides(select, _IKQueue)

@implementer(IReactorFDSet, IReactorDaemonize)
class KQueueReactor(posixbase.PosixReactorBase):
    """
    A reactor that uses kqueue(2)/kevent(2) and relies on Python 2.6 or higher
    which has built in support for kqueue in the select module.

    @ivar _kq: A C{kqueue} which will be used to check for I/O readiness.

    @ivar _impl: The implementation of L{_IKQueue} to use.

    @ivar _selectables: A dictionary mapping integer file descriptors to
        instances of L{FileDescriptor} which have been registered with the
        reactor.  All L{FileDescriptor}s which are currently receiving read or
        write readiness notifications will be present as values in this
        dictionary.

    @ivar _reads: A set containing integer file descriptors.  Values in this
        set will be registered with C{_kq} for read readiness notifications
        which will be dispatched to the corresponding L{FileDescriptor}
        instances in C{_selectables}.

    @ivar _writes: A set containing integer file descriptors.  Values in this
        set will be registered with C{_kq} for write readiness notifications
        which will be dispatched to the corresponding L{FileDescriptor}
        instances in C{_selectables}.
    """

    def __init__(self, _kqueueImpl=select):
        if False:
            while True:
                i = 10
        '\n        Initialize kqueue object, file descriptor tracking dictionaries, and\n        the base class.\n\n        See:\n            - http://docs.python.org/library/select.html\n            - www.freebsd.org/cgi/man.cgi?query=kqueue\n            - people.freebsd.org/~jlemon/papers/kqueue.pdf\n\n        @param _kqueueImpl: The implementation of L{_IKQueue} to use. A\n            hook for testing.\n        '
        self._impl = _kqueueImpl
        self._kq = self._impl.kqueue()
        self._reads = set()
        self._writes = set()
        self._selectables = {}
        posixbase.PosixReactorBase.__init__(self)

    def _updateRegistration(self, fd, filter, op):
        if False:
            i = 10
            return i + 15
        '\n        Private method for changing kqueue registration on a given FD\n        filtering for events given filter/op. This will never block and\n        returns nothing.\n        '
        self._kq.control([self._impl.kevent(fd, filter, op)], 0, 0)

    def beforeDaemonize(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement L{IReactorDaemonize.beforeDaemonize}.\n        '
        self._kq.close()
        self._kq = None

    def afterDaemonize(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorDaemonize.afterDaemonize}.\n        '
        self._kq = self._impl.kqueue()
        for fd in self._reads:
            self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_ADD)
        for fd in self._writes:
            self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_ADD)

    def addReader(self, reader):
        if False:
            print('Hello World!')
        '\n        Implement L{IReactorFDSet.addReader}.\n        '
        fd = reader.fileno()
        if fd not in self._reads:
            try:
                self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_ADD)
            except OSError:
                pass
            finally:
                self._selectables[fd] = reader
                self._reads.add(fd)

    def addWriter(self, writer):
        if False:
            return 10
        '\n        Implement L{IReactorFDSet.addWriter}.\n        '
        fd = writer.fileno()
        if fd not in self._writes:
            try:
                self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_ADD)
            except OSError:
                pass
            finally:
                self._selectables[fd] = writer
                self._writes.add(fd)

    def removeReader(self, reader):
        if False:
            i = 10
            return i + 15
        '\n        Implement L{IReactorFDSet.removeReader}.\n        '
        wasLost = False
        try:
            fd = reader.fileno()
        except BaseException:
            fd = -1
        if fd == -1:
            for (fd, fdes) in self._selectables.items():
                if reader is fdes:
                    wasLost = True
                    break
            else:
                return
        if fd in self._reads:
            self._reads.remove(fd)
            if fd not in self._writes:
                del self._selectables[fd]
            if not wasLost:
                try:
                    self._updateRegistration(fd, KQ_FILTER_READ, KQ_EV_DELETE)
                except OSError:
                    pass

    def removeWriter(self, writer):
        if False:
            i = 10
            return i + 15
        '\n        Implement L{IReactorFDSet.removeWriter}.\n        '
        wasLost = False
        try:
            fd = writer.fileno()
        except BaseException:
            fd = -1
        if fd == -1:
            for (fd, fdes) in self._selectables.items():
                if writer is fdes:
                    wasLost = True
                    break
            else:
                return
        if fd in self._writes:
            self._writes.remove(fd)
            if fd not in self._reads:
                del self._selectables[fd]
            if not wasLost:
                try:
                    self._updateRegistration(fd, KQ_FILTER_WRITE, KQ_EV_DELETE)
                except OSError:
                    pass

    def removeAll(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorFDSet.removeAll}.\n        '
        return self._removeAll([self._selectables[fd] for fd in self._reads], [self._selectables[fd] for fd in self._writes])

    def getReaders(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorFDSet.getReaders}.\n        '
        return [self._selectables[fd] for fd in self._reads]

    def getWriters(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorFDSet.getWriters}.\n        '
        return [self._selectables[fd] for fd in self._writes]

    def doKEvent(self, timeout):
        if False:
            return 10
        '\n        Poll the kqueue for new events.\n        '
        if timeout is None:
            timeout = 1
        try:
            events = self._kq.control([], len(self._selectables), timeout)
        except OSError as e:
            if e.errno == errno.EINTR:
                return
            else:
                raise
        _drdw = self._doWriteOrRead
        for event in events:
            fd = event.ident
            try:
                selectable = self._selectables[fd]
            except KeyError:
                continue
            else:
                log.callWithLogger(selectable, _drdw, selectable, fd, event)

    def _doWriteOrRead(self, selectable, fd, event):
        if False:
            while True:
                i = 10
        '\n        Private method called when a FD is ready for reading, writing or was\n        lost. Do the work and raise errors where necessary.\n        '
        why = None
        inRead = False
        (filter, flags, data, fflags) = (event.filter, event.flags, event.data, event.fflags)
        if flags & KQ_EV_EOF and data and fflags:
            why = main.CONNECTION_LOST
        else:
            try:
                if selectable.fileno() == -1:
                    inRead = False
                    why = posixbase._NO_FILEDESC
                else:
                    if filter == KQ_FILTER_READ:
                        inRead = True
                        why = selectable.doRead()
                    if filter == KQ_FILTER_WRITE:
                        inRead = False
                        why = selectable.doWrite()
            except BaseException:
                why = failure.Failure()
                log.err(why, 'An exception was raised from application code while processing a reactor selectable')
        if why:
            self._disconnectSelectable(selectable, why, inRead)
    doIteration = doKEvent

def install():
    if False:
        while True:
            i = 10
    '\n    Install the kqueue() reactor.\n    '
    p = KQueueReactor()
    from twisted.internet.main import installReactor
    installReactor(p)
__all__ = ['KQueueReactor', 'install']