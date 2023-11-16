"""
asyncio-based reactor implementation.
"""
import errno
import sys
from asyncio import AbstractEventLoop, get_event_loop
from typing import Dict, Optional, Type
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase, _ContinuousPolling
from twisted.logger import Logger
from twisted.python.log import callWithLogger

@implementer(IReactorFDSet)
class AsyncioSelectorReactor(PosixReactorBase):
    """
    Reactor running on top of L{asyncio.SelectorEventLoop}.

    On POSIX platforms, the default event loop is
    L{asyncio.SelectorEventLoop}.
    On Windows, the default event loop on Python 3.7 and older
    is C{asyncio.WindowsSelectorEventLoop}, but on Python 3.8 and newer
    the default event loop is C{asyncio.WindowsProactorEventLoop} which
    is incompatible with L{AsyncioSelectorReactor}.
    Applications that use L{AsyncioSelectorReactor} on Windows
    with Python 3.8+ must call
    C{asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())}
    before instantiating and running L{AsyncioSelectorReactor}.
    """
    _asyncClosed = False
    _log = Logger()

    def __init__(self, eventloop: Optional[AbstractEventLoop]=None):
        if False:
            i = 10
            return i + 15
        if eventloop is None:
            _eventloop: AbstractEventLoop = get_event_loop()
        else:
            _eventloop = eventloop
        if sys.platform == 'win32':
            from asyncio import ProactorEventLoop
            if isinstance(_eventloop, ProactorEventLoop):
                raise TypeError(f'ProactorEventLoop is not supported, got: {_eventloop}')
        self._asyncioEventloop: AbstractEventLoop = _eventloop
        self._writers: Dict[Type[FileDescriptor], int] = {}
        self._readers: Dict[Type[FileDescriptor], int] = {}
        self._continuousPolling = _ContinuousPolling(self)
        self._scheduledAt = None
        self._timerHandle = None
        super().__init__()

    def _unregisterFDInAsyncio(self, fd):
        if False:
            i = 10
            return i + 15
        "\n        Compensate for a bug in asyncio where it will not unregister a FD that\n        it cannot handle in the epoll loop. It touches internal asyncio code.\n\n        A description of the bug by markrwilliams:\n\n        The C{add_writer} method of asyncio event loops isn't atomic because\n        all the Selector classes in the selector module internally record a\n        file object before passing it to the platform's selector\n        implementation. If the platform's selector decides the file object\n        isn't acceptable, the resulting exception doesn't cause the Selector to\n        un-track the file object.\n\n        The failing/hanging stdio test goes through the following sequence of\n        events (roughly):\n\n        * The first C{connection.write(intToByte(value))} call hits the asyncio\n        reactor's C{addWriter} method.\n\n        * C{addWriter} calls the asyncio loop's C{add_writer} method, which\n        happens to live on C{_BaseSelectorEventLoop}.\n\n        * The asyncio loop's C{add_writer} method checks if the file object has\n        been registered before via the selector's C{get_key} method.\n\n        * It hasn't, so the KeyError block runs and calls the selector's\n        register method\n\n        * Code examples that follow use EpollSelector, but the code flow holds\n        true for any other selector implementation. The selector's register\n        method first calls through to the next register method in the MRO\n\n        * That next method is always C{_BaseSelectorImpl.register} which\n        creates a C{SelectorKey} instance for the file object, stores it under\n        the file object's file descriptor, and then returns it.\n\n        * Control returns to the concrete selector implementation, which asks\n        the operating system to track the file descriptor using the right API.\n\n        * The operating system refuses! An exception is raised that, in this\n        case, the asyncio reactor handles by creating a C{_ContinuousPolling}\n        object to watch the file descriptor.\n\n        * The second C{connection.write(intToByte(value))} call hits the\n        asyncio reactor's C{addWriter} method, which hits the C{add_writer}\n        method. But the loop's selector's get_key method now returns a\n        C{SelectorKey}! Now the asyncio reactor's C{addWriter} method thinks\n        the asyncio loop will watch the file descriptor, even though it won't.\n        "
        try:
            self._asyncioEventloop._selector.unregister(fd)
        except BaseException:
            pass

    def _readOrWrite(self, selectable, read):
        if False:
            while True:
                i = 10
        method = selectable.doRead if read else selectable.doWrite
        if selectable.fileno() == -1:
            self._disconnectSelectable(selectable, _NO_FILEDESC, read)
            return
        try:
            why = method()
        except Exception as e:
            why = e
            self._log.failure(None)
        if why:
            self._disconnectSelectable(selectable, why, read)

    def addReader(self, reader):
        if False:
            for i in range(10):
                print('nop')
        if reader in self._readers.keys() or reader in self._continuousPolling._readers:
            return
        fd = reader.fileno()
        try:
            self._asyncioEventloop.add_reader(fd, callWithLogger, reader, self._readOrWrite, reader, True)
            self._readers[reader] = fd
        except OSError as e:
            self._unregisterFDInAsyncio(fd)
            if e.errno == errno.EPERM:
                self._continuousPolling.addReader(reader)
            else:
                raise

    def addWriter(self, writer):
        if False:
            print('Hello World!')
        if writer in self._writers.keys() or writer in self._continuousPolling._writers:
            return
        fd = writer.fileno()
        try:
            self._asyncioEventloop.add_writer(fd, callWithLogger, writer, self._readOrWrite, writer, False)
            self._writers[writer] = fd
        except PermissionError:
            self._unregisterFDInAsyncio(fd)
            self._continuousPolling.addWriter(writer)
        except BrokenPipeError:
            self._unregisterFDInAsyncio(fd)
        except BaseException:
            self._unregisterFDInAsyncio(fd)
            raise

    def removeReader(self, reader):
        if False:
            return 10
        if not (reader in self._readers.keys() or self._continuousPolling.isReading(reader)):
            return
        if self._continuousPolling.isReading(reader):
            self._continuousPolling.removeReader(reader)
            return
        fd = reader.fileno()
        if fd == -1:
            fd = self._readers.pop(reader)
        else:
            self._readers.pop(reader)
        self._asyncioEventloop.remove_reader(fd)

    def removeWriter(self, writer):
        if False:
            print('Hello World!')
        if not (writer in self._writers.keys() or self._continuousPolling.isWriting(writer)):
            return
        if self._continuousPolling.isWriting(writer):
            self._continuousPolling.removeWriter(writer)
            return
        fd = writer.fileno()
        if fd == -1:
            fd = self._writers.pop(writer)
        else:
            self._writers.pop(writer)
        self._asyncioEventloop.remove_writer(fd)

    def removeAll(self):
        if False:
            i = 10
            return i + 15
        return self._removeAll(self._readers.keys(), self._writers.keys()) + self._continuousPolling.removeAll()

    def getReaders(self):
        if False:
            while True:
                i = 10
        return list(self._readers.keys()) + self._continuousPolling.getReaders()

    def getWriters(self):
        if False:
            print('Hello World!')
        return list(self._writers.keys()) + self._continuousPolling.getWriters()

    def iterate(self, timeout):
        if False:
            print('Hello World!')
        self._asyncioEventloop.call_later(timeout + 0.01, self._asyncioEventloop.stop)
        self._asyncioEventloop.run_forever()

    def run(self, installSignalHandlers=True):
        if False:
            while True:
                i = 10
        self.startRunning(installSignalHandlers=installSignalHandlers)
        self._asyncioEventloop.run_forever()
        if self._justStopped:
            self._justStopped = False

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        super().stop()
        self.callLater(0, lambda : None)

    def crash(self):
        if False:
            while True:
                i = 10
        super().crash()
        self._asyncioEventloop.stop()

    def _onTimer(self):
        if False:
            i = 10
            return i + 15
        self._scheduledAt = None
        self.runUntilCurrent()
        self._reschedule()

    def _reschedule(self):
        if False:
            return 10
        timeout = self.timeout()
        if timeout is not None:
            abs_time = self._asyncioEventloop.time() + timeout
            self._scheduledAt = abs_time
            if self._timerHandle is not None:
                self._timerHandle.cancel()
            self._timerHandle = self._asyncioEventloop.call_at(abs_time, self._onTimer)

    def _moveCallLaterSooner(self, tple):
        if False:
            while True:
                i = 10
        PosixReactorBase._moveCallLaterSooner(self, tple)
        self._reschedule()

    def callLater(self, seconds, f, *args, **kwargs):
        if False:
            print('Hello World!')
        dc = PosixReactorBase.callLater(self, seconds, f, *args, **kwargs)
        abs_time = self._asyncioEventloop.time() + self.timeout()
        if self._scheduledAt is None or abs_time < self._scheduledAt:
            self._reschedule()
        return dc

    def callFromThread(self, f, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        g = lambda : self.callLater(0, f, *args, **kwargs)
        self._asyncioEventloop.call_soon_threadsafe(g)

def install(eventloop=None):
    if False:
        print('Hello World!')
    '\n    Install an asyncio-based reactor.\n\n    @param eventloop: The asyncio eventloop to wrap. If default, the global one\n        is selected.\n    '
    reactor = AsyncioSelectorReactor(eventloop)
    from twisted.internet.main import installReactor
    installReactor(reactor)