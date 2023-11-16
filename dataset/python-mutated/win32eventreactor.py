"""
A win32event based implementation of the Twisted main loop.

This requires pywin32 (formerly win32all) or ActivePython to be installed.

To install the event loop (and you should do this before any connections,
listeners or connectors are added)::

    from twisted.internet import win32eventreactor
    win32eventreactor.install()

LIMITATIONS:
 1. WaitForMultipleObjects and thus the event loop can only handle 64 objects.
 2. Process running has some problems (see L{twisted.internet.process} docstring).


TODO:
 1. Event loop handling of writes is *very* problematic (this is causing failed tests).
    Switch to doing it the correct way, whatever that means (see below).
 2. Replace icky socket loopback waker with event based waker (use dummyEvent object)
 3. Switch everyone to using Free Software so we don't have to deal with proprietary APIs.


ALTERNATIVE SOLUTIONS:
 - IIRC, sockets can only be registered once. So we switch to a structure
   like the poll() reactor, thus allowing us to deal with write events in
   a decent fashion. This should allow us to pass tests, but we're still
   limited to 64 events.

Or:

 - Instead of doing a reactor, we make this an addon to the select reactor.
   The WFMO event loop runs in a separate thread. This means no need to maintain
   separate code for networking, 64 event limit doesn't apply to sockets,
   we can run processes and other win32 stuff in default event loop. The
   only problem is that we're stuck with the icky socket based waker.
   Another benefit is that this could be extended to support >64 events
   in a simpler manner than the previous solution.

The 2nd solution is probably what will get implemented.
"""
import sys
import time
from threading import Thread
from weakref import WeakKeyDictionary
from zope.interface import implementer
from win32file import FD_ACCEPT, FD_CLOSE, FD_CONNECT, FD_READ, WSAEventSelect
try:
    from win32file import WSAEnumNetworkEvents
except ImportError:
    import warnings
    warnings.warn('Reliable disconnection notification requires pywin32 215 or later', category=UserWarning)

    def WSAEnumNetworkEvents(fd, event):
        if False:
            print('Hello World!')
        return {FD_READ}
import win32gui
from win32event import QS_ALLINPUT, WAIT_OBJECT_0, WAIT_TIMEOUT, CreateEvent, MsgWaitForMultipleObjects
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet, IReactorWin32Events
from twisted.internet.threads import blockingCallFromThread
from twisted.python import failure, log, threadable

@implementer(IReactorFDSet, IReactorWin32Events)
class Win32Reactor(posixbase.PosixReactorBase):
    """
    Reactor that uses Win32 event APIs.

    @ivar _reads: A dictionary mapping L{FileDescriptor} instances to a
        win32 event object used to check for read events for that descriptor.

    @ivar _writes: A dictionary mapping L{FileDescriptor} instances to a
        arbitrary value.  Keys in this dictionary will be given a chance to
        write out their data.

    @ivar _events: A dictionary mapping win32 event object to tuples of
        L{FileDescriptor} instances and event masks.

    @ivar _closedAndReading: Along with C{_closedAndNotReading}, keeps track of
        descriptors which have had close notification delivered from the OS but
        which we have not finished reading data from.  MsgWaitForMultipleObjects
        will only deliver close notification to us once, so we remember it in
        these two dictionaries until we're ready to act on it.  The OS has
        delivered close notification for each descriptor in this dictionary, and
        the descriptors are marked as allowed to handle read events in the
        reactor, so they can be processed.  When a descriptor is marked as not
        allowed to handle read events in the reactor (ie, it is passed to
        L{IReactorFDSet.removeReader}), it is moved out of this dictionary and
        into C{_closedAndNotReading}.  The descriptors are keys in this
        dictionary.  The values are arbitrary.
    @type _closedAndReading: C{dict}

    @ivar _closedAndNotReading: These descriptors have had close notification
        delivered from the OS, but are not marked as allowed to handle read
        events in the reactor.  They are saved here to record their closed
        state, but not processed at all.  When one of these descriptors is
        passed to L{IReactorFDSet.addReader}, it is moved out of this dictionary
        and into C{_closedAndReading}.  The descriptors are keys in this
        dictionary.  The values are arbitrary.  This is a weak key dictionary so
        that if an application tells the reactor to stop reading from a
        descriptor and then forgets about that descriptor itself, the reactor
        will also forget about it.
    @type _closedAndNotReading: C{WeakKeyDictionary}
    """
    dummyEvent = CreateEvent(None, 0, 0, None)

    def __init__(self):
        if False:
            print('Hello World!')
        self._reads = {}
        self._writes = {}
        self._events = {}
        self._closedAndReading = {}
        self._closedAndNotReading = WeakKeyDictionary()
        posixbase.PosixReactorBase.__init__(self)

    def _makeSocketEvent(self, fd, action, why):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a win32 event object for a socket.\n        '
        event = CreateEvent(None, 0, 0, None)
        WSAEventSelect(fd, event, why)
        self._events[event] = (fd, action)
        return event

    def addEvent(self, event, fd, action):
        if False:
            i = 10
            return i + 15
        '\n        Add a new win32 event to the event loop.\n        '
        self._events[event] = (fd, action)

    def removeEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        Remove an event.\n        '
        del self._events[event]

    def addReader(self, reader):
        if False:
            while True:
                i = 10
        '\n        Add a socket FileDescriptor for notification of data available to read.\n        '
        if reader not in self._reads:
            self._reads[reader] = self._makeSocketEvent(reader, 'doRead', FD_READ | FD_ACCEPT | FD_CONNECT | FD_CLOSE)
            if reader in self._closedAndNotReading:
                self._closedAndReading[reader] = True
                del self._closedAndNotReading[reader]

    def addWriter(self, writer):
        if False:
            i = 10
            return i + 15
        '\n        Add a socket FileDescriptor for notification of data available to write.\n        '
        if writer not in self._writes:
            self._writes[writer] = 1

    def removeReader(self, reader):
        if False:
            return 10
        'Remove a Selectable for notification of data available to read.'
        if reader in self._reads:
            del self._events[self._reads[reader]]
            del self._reads[reader]
            if reader in self._closedAndReading:
                self._closedAndNotReading[reader] = True
                del self._closedAndReading[reader]

    def removeWriter(self, writer):
        if False:
            i = 10
            return i + 15
        'Remove a Selectable for notification of data available to write.'
        if writer in self._writes:
            del self._writes[writer]

    def removeAll(self):
        if False:
            return 10
        '\n        Remove all selectables, and return a list of them.\n        '
        return self._removeAll(self._reads, self._writes)

    def getReaders(self):
        if False:
            while True:
                i = 10
        return list(self._reads.keys())

    def getWriters(self):
        if False:
            while True:
                i = 10
        return list(self._writes.keys())

    def doWaitForMultipleEvents(self, timeout):
        if False:
            return 10
        log.msg(channel='system', event='iteration', reactor=self)
        if timeout is None:
            timeout = 100
        ranUserCode = False
        for reader in list(self._closedAndReading.keys()):
            ranUserCode = True
            self._runAction('doRead', reader)
        for fd in list(self._writes.keys()):
            ranUserCode = True
            log.callWithLogger(fd, self._runWrite, fd)
        if ranUserCode:
            timeout = 0
        if not (self._events or self._writes):
            time.sleep(timeout)
            return
        handles = list(self._events.keys()) or [self.dummyEvent]
        timeout = int(timeout * 1000)
        val = MsgWaitForMultipleObjects(handles, 0, timeout, QS_ALLINPUT)
        if val == WAIT_TIMEOUT:
            return
        elif val == WAIT_OBJECT_0 + len(handles):
            exit = win32gui.PumpWaitingMessages()
            if exit:
                self.callLater(0, self.stop)
                return
        elif val >= WAIT_OBJECT_0 and val < WAIT_OBJECT_0 + len(handles):
            event = handles[val - WAIT_OBJECT_0]
            (fd, action) = self._events[event]
            if fd in self._reads:
                fileno = fd.fileno()
                if fileno == -1:
                    self._disconnectSelectable(fd, posixbase._NO_FILEDESC, False)
                    return
                events = WSAEnumNetworkEvents(fileno, event)
                if FD_CLOSE in events:
                    self._closedAndReading[fd] = True
            log.callWithLogger(fd, self._runAction, action, fd)

    def _runWrite(self, fd):
        if False:
            i = 10
            return i + 15
        closed = 0
        try:
            closed = fd.doWrite()
        except BaseException:
            closed = sys.exc_info()[1]
            log.deferr()
        if closed:
            self.removeReader(fd)
            self.removeWriter(fd)
            try:
                fd.connectionLost(failure.Failure(closed))
            except BaseException:
                log.deferr()
        elif closed is None:
            return 1

    def _runAction(self, action, fd):
        if False:
            i = 10
            return i + 15
        try:
            closed = getattr(fd, action)()
        except BaseException:
            closed = sys.exc_info()[1]
            log.deferr()
        if closed:
            self._disconnectSelectable(fd, closed, action == 'doRead')
    doIteration = doWaitForMultipleEvents

class _ThreadFDWrapper:
    """
    This wraps an event handler and translates notification in the helper
    L{Win32Reactor} thread into a notification in the primary reactor thread.

    @ivar _reactor: The primary reactor, the one to which event notification
        will be sent.

    @ivar _fd: The L{FileDescriptor} to which the event will be dispatched.

    @ivar _action: A C{str} giving the method of C{_fd} which handles the event.

    @ivar _logPrefix: The pre-fetched log prefix string for C{_fd}, so that
        C{_fd.logPrefix} does not need to be called in a non-main thread.
    """

    def __init__(self, reactor, fd, action, logPrefix):
        if False:
            i = 10
            return i + 15
        self._reactor = reactor
        self._fd = fd
        self._action = action
        self._logPrefix = logPrefix

    def logPrefix(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the original handler's log prefix, as it was given to\n        C{__init__}.\n        "
        return self._logPrefix

    def _execute(self):
        if False:
            return 10
        '\n        Callback fired when the associated event is set.  Run the C{action}\n        callback on the wrapped descriptor in the main reactor thread and raise\n        or return whatever it raises or returns to cause this event handler to\n        be removed from C{self._reactor} if appropriate.\n        '
        return blockingCallFromThread(self._reactor, lambda : getattr(self._fd, self._action)())

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pass through to the wrapped descriptor, but in the main reactor thread\n        instead of the helper C{Win32Reactor} thread.\n        '
        self._reactor.callFromThread(self._fd.connectionLost, reason)

@implementer(IReactorWin32Events)
class _ThreadedWin32EventsMixin:
    """
    This mixin implements L{IReactorWin32Events} for another reactor by running
    a L{Win32Reactor} in a separate thread and dispatching work to it.

    @ivar _reactor: The L{Win32Reactor} running in the other thread.  This is
        L{None} until it is actually needed.

    @ivar _reactorThread: The L{threading.Thread} which is running the
        L{Win32Reactor}.  This is L{None} until it is actually needed.
    """
    _reactor = None
    _reactorThread = None

    def _unmakeHelperReactor(self):
        if False:
            i = 10
            return i + 15
        '\n        Stop and discard the reactor started by C{_makeHelperReactor}.\n        '
        self._reactor.callFromThread(self._reactor.stop)
        self._reactor = None

    def _makeHelperReactor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create and (in a new thread) start a L{Win32Reactor} instance to use for\n        the implementation of L{IReactorWin32Events}.\n        '
        self._reactor = Win32Reactor()
        self._reactor._registerAsIOThread = False
        self._reactorThread = Thread(target=self._reactor.run, args=(False,))
        self.addSystemEventTrigger('after', 'shutdown', self._unmakeHelperReactor)
        self._reactorThread.start()

    def addEvent(self, event, fd, action):
        if False:
            print('Hello World!')
        '\n        @see: L{IReactorWin32Events}\n        '
        if self._reactor is None:
            self._makeHelperReactor()
        self._reactor.callFromThread(self._reactor.addEvent, event, _ThreadFDWrapper(self, fd, action, fd.logPrefix()), '_execute')

    def removeEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        @see: L{IReactorWin32Events}\n        '
        self._reactor.callFromThread(self._reactor.removeEvent, event)

def install():
    if False:
        return 10
    threadable.init(1)
    r = Win32Reactor()
    from . import main
    main.installReactor(r)
__all__ = ['Win32Reactor', 'install']