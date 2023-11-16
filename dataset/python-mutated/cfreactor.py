"""
A reactor for integrating with U{CFRunLoop<http://bit.ly/cfrunloop>}, the
CoreFoundation main loop used by macOS.

This is useful for integrating Twisted with U{PyObjC<http://pyobjc.sf.net/>}
applications.
"""
from __future__ import annotations
__all__ = ['install', 'CFReactor']
import sys
from zope.interface import implementer
from CFNetwork import CFSocketCreateRunLoopSource, CFSocketCreateWithNative, CFSocketDisableCallBacks, CFSocketEnableCallBacks, CFSocketInvalidate, CFSocketSetSocketFlags, kCFSocketAutomaticallyReenableReadCallBack, kCFSocketAutomaticallyReenableWriteCallBack, kCFSocketConnectCallBack, kCFSocketReadCallBack, kCFSocketWriteCallBack
from CoreFoundation import CFAbsoluteTimeGetCurrent, CFRunLoopAddSource, CFRunLoopAddTimer, CFRunLoopGetCurrent, CFRunLoopRemoveSource, CFRunLoopRun, CFRunLoopStop, CFRunLoopTimerCreate, CFRunLoopTimerInvalidate, kCFAllocatorDefault, kCFRunLoopCommonModes
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
_READ = 0
_WRITE = 1
_preserveSOError = 1 << 6

class _WakerPlus(_UnixWaker):
    """
    The normal Twisted waker will simply wake up the main loop, which causes an
    iteration to run, which in turn causes L{ReactorBase.runUntilCurrent}
    to get invoked.

    L{CFReactor} has a slightly different model of iteration, though: rather
    than have each iteration process the thread queue, then timed calls, then
    file descriptors, each callback is run as it is dispatched by the CFRunLoop
    observer which triggered it.

    So this waker needs to not only unblock the loop, but also make sure the
    work gets done; so, it reschedules the invocation of C{runUntilCurrent} to
    be immediate (0 seconds from now) even if there is no timed call work to
    do.
    """

    def __init__(self, reactor):
        if False:
            while True:
                i = 10
        super().__init__()
        self.reactor = reactor

    def doRead(self):
        if False:
            print('Hello World!')
        '\n        Wake up the loop and force C{runUntilCurrent} to run immediately in the\n        next timed iteration.\n        '
        result = super().doRead()
        self.reactor._scheduleSimulate(True)
        return result

@implementer(IReactorFDSet)
class CFReactor(PosixReactorBase):
    """
    The CoreFoundation reactor.

    You probably want to use this via the L{install} API.

    @ivar _fdmap: a dictionary, mapping an integer (a file descriptor) to a
        4-tuple of:

            - source: a C{CFRunLoopSource}; the source associated with this
              socket.
            - socket: a C{CFSocket} wrapping the file descriptor.
            - descriptor: an L{IReadDescriptor} and/or L{IWriteDescriptor}
              provider.
            - read-write: a 2-C{list} of booleans: respectively, whether this
              descriptor is currently registered for reading or registered for
              writing.

    @ivar _idmap: a dictionary, mapping the id() of an L{IReadDescriptor} or
        L{IWriteDescriptor} to a C{fd} in L{_fdmap}.  Implemented in this
        manner so that we don't have to rely (even more) on the hashability of
        L{IReadDescriptor} providers, and we know that they won't be collected
        since these are kept in sync with C{_fdmap}.  Necessary because the
        .fileno() of a file descriptor may change at will, so we need to be
        able to look up what its file descriptor I{used} to be, so that we can
        look it up in C{_fdmap}

    @ivar _cfrunloop: the C{CFRunLoop} pyobjc object wrapped
        by this reactor.

    @ivar _inCFLoop: Is C{CFRunLoopRun} currently running?

    @type _inCFLoop: L{bool}

    @ivar _currentSimulator: if a CFTimer is currently scheduled with the CF
        run loop to run Twisted callLater calls, this is a reference to it.
        Otherwise, it is L{None}
    """

    def __init__(self, runLoop=None, runner=None):
        if False:
            i = 10
            return i + 15
        self._fdmap = {}
        self._idmap = {}
        if runner is None:
            runner = CFRunLoopRun
        self._runner = runner
        if runLoop is None:
            runLoop = CFRunLoopGetCurrent()
        self._cfrunloop = runLoop
        PosixReactorBase.__init__(self)

    def _wakerFactory(self) -> _WakerPlus:
        if False:
            return 10
        return _WakerPlus(self)

    def _socketCallback(self, cfSocket, callbackType, ignoredAddress, ignoredData, context):
        if False:
            while True:
                i = 10
        '\n        The socket callback issued by CFRunLoop.  This will issue C{doRead} or\n        C{doWrite} calls to the L{IReadDescriptor} and L{IWriteDescriptor}\n        registered with the file descriptor that we are being notified of.\n\n        @param cfSocket: The C{CFSocket} which has got some activity.\n\n        @param callbackType: The type of activity that we are being notified\n            of.  Either C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}.\n\n        @param ignoredAddress: Unused, because this is not used for either of\n            the callback types we register for.\n\n        @param ignoredData: Unused, because this is not used for either of the\n            callback types we register for.\n\n        @param context: The data associated with this callback by\n            C{CFSocketCreateWithNative} (in C{CFReactor._watchFD}).  A 2-tuple\n            of C{(int, CFRunLoopSource)}.\n        '
        (fd, smugglesrc) = context
        if fd not in self._fdmap:
            CFRunLoopRemoveSource(self._cfrunloop, smugglesrc, kCFRunLoopCommonModes)
            return
        (src, skt, readWriteDescriptor, rw) = self._fdmap[fd]

        def _drdw():
            if False:
                i = 10
                return i + 15
            why = None
            isRead = False
            try:
                if readWriteDescriptor.fileno() == -1:
                    why = _NO_FILEDESC
                else:
                    isRead = callbackType == kCFSocketReadCallBack
                    if isRead:
                        if rw[_READ]:
                            why = readWriteDescriptor.doRead()
                    elif rw[_WRITE]:
                        why = readWriteDescriptor.doWrite()
            except BaseException:
                why = sys.exc_info()[1]
                log.err()
            if why:
                self._disconnectSelectable(readWriteDescriptor, why, isRead)
        log.callWithLogger(readWriteDescriptor, _drdw)

    def _watchFD(self, fd, descr, flag):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register a file descriptor with the C{CFRunLoop}, or modify its state\n        so that it's listening for both notifications (read and write) rather\n        than just one; used to implement C{addReader} and C{addWriter}.\n\n        @param fd: The file descriptor.\n\n        @type fd: L{int}\n\n        @param descr: the L{IReadDescriptor} or L{IWriteDescriptor}\n\n        @param flag: the flag to register for callbacks on, either\n            C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}\n        "
        if fd == -1:
            raise RuntimeError('Invalid file descriptor.')
        if fd in self._fdmap:
            (src, cfs, gotdescr, rw) = self._fdmap[fd]
        else:
            ctx = []
            ctx.append(fd)
            cfs = CFSocketCreateWithNative(kCFAllocatorDefault, fd, kCFSocketReadCallBack | kCFSocketWriteCallBack | kCFSocketConnectCallBack, self._socketCallback, ctx)
            CFSocketSetSocketFlags(cfs, kCFSocketAutomaticallyReenableReadCallBack | kCFSocketAutomaticallyReenableWriteCallBack | _preserveSOError)
            src = CFSocketCreateRunLoopSource(kCFAllocatorDefault, cfs, 0)
            ctx.append(src)
            CFRunLoopAddSource(self._cfrunloop, src, kCFRunLoopCommonModes)
            CFSocketDisableCallBacks(cfs, kCFSocketReadCallBack | kCFSocketWriteCallBack | kCFSocketConnectCallBack)
            rw = [False, False]
            self._idmap[id(descr)] = fd
            self._fdmap[fd] = (src, cfs, descr, rw)
        rw[self._flag2idx(flag)] = True
        CFSocketEnableCallBacks(cfs, flag)

    def _flag2idx(self, flag):
        if False:
            print('Hello World!')
        '\n        Convert a C{kCFSocket...} constant to an index into the read/write\n        state list (C{_READ} or C{_WRITE}) (the 4th element of the value of\n        C{self._fdmap}).\n\n        @param flag: C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}\n\n        @return: C{_READ} or C{_WRITE}\n        '
        return {kCFSocketReadCallBack: _READ, kCFSocketWriteCallBack: _WRITE}[flag]

    def _unwatchFD(self, fd, descr, flag):
        if False:
            print('Hello World!')
        "\n        Unregister a file descriptor with the C{CFRunLoop}, or modify its state\n        so that it's listening for only one notification (read or write) as\n        opposed to both; used to implement C{removeReader} and C{removeWriter}.\n\n        @param fd: a file descriptor\n\n        @type fd: C{int}\n\n        @param descr: an L{IReadDescriptor} or L{IWriteDescriptor}\n\n        @param flag: C{kCFSocketWriteCallBack} C{kCFSocketReadCallBack}\n        "
        if id(descr) not in self._idmap:
            return
        if fd == -1:
            realfd = self._idmap[id(descr)]
        else:
            realfd = fd
        (src, cfs, descr, rw) = self._fdmap[realfd]
        CFSocketDisableCallBacks(cfs, flag)
        rw[self._flag2idx(flag)] = False
        if not rw[_READ] and (not rw[_WRITE]):
            del self._idmap[id(descr)]
            del self._fdmap[realfd]
            CFRunLoopRemoveSource(self._cfrunloop, src, kCFRunLoopCommonModes)
            CFSocketInvalidate(cfs)

    def addReader(self, reader):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement L{IReactorFDSet.addReader}.\n        '
        self._watchFD(reader.fileno(), reader, kCFSocketReadCallBack)

    def addWriter(self, writer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement L{IReactorFDSet.addWriter}.\n        '
        self._watchFD(writer.fileno(), writer, kCFSocketWriteCallBack)

    def removeReader(self, reader):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorFDSet.removeReader}.\n        '
        self._unwatchFD(reader.fileno(), reader, kCFSocketReadCallBack)

    def removeWriter(self, writer):
        if False:
            print('Hello World!')
        '\n        Implement L{IReactorFDSet.removeWriter}.\n        '
        self._unwatchFD(writer.fileno(), writer, kCFSocketWriteCallBack)

    def removeAll(self):
        if False:
            i = 10
            return i + 15
        '\n        Implement L{IReactorFDSet.removeAll}.\n        '
        allDesc = {descr for (src, cfs, descr, rw) in self._fdmap.values()}
        allDesc -= set(self._internalReaders)
        for desc in allDesc:
            self.removeReader(desc)
            self.removeWriter(desc)
        return list(allDesc)

    def getReaders(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement L{IReactorFDSet.getReaders}.\n        '
        return [descr for (src, cfs, descr, rw) in self._fdmap.values() if rw[_READ]]

    def getWriters(self):
        if False:
            print('Hello World!')
        '\n        Implement L{IReactorFDSet.getWriters}.\n        '
        return [descr for (src, cfs, descr, rw) in self._fdmap.values() if rw[_WRITE]]

    def _moveCallLaterSooner(self, tple):
        if False:
            i = 10
            return i + 15
        "\n        Override L{PosixReactorBase}'s implementation of L{IDelayedCall.reset}\n        so that it will immediately reschedule.  Normally\n        C{_moveCallLaterSooner} depends on the fact that C{runUntilCurrent} is\n        always run before the mainloop goes back to sleep, so this forces it to\n        immediately recompute how long the loop needs to stay asleep.\n        "
        result = PosixReactorBase._moveCallLaterSooner(self, tple)
        self._scheduleSimulate()
        return result

    def startRunning(self, installSignalHandlers: bool=True) -> None:
        if False:
            return 10
        "\n        Start running the reactor, then kick off the timer that advances\n        Twisted's clock to keep pace with CFRunLoop's.\n        "
        super().startRunning(installSignalHandlers)
        self._scheduleSimulate(force=True)
    _inCFLoop = False

    def mainLoop(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Run the runner (C{CFRunLoopRun} or something that calls it), which runs\n        the run loop until C{crash()} is called.\n        '
        if not self._started:

            def docrash() -> None:
                if False:
                    print('Hello World!')
                self.crash()
            self._started = True
            self.callLater(0, docrash)
        already = False
        try:
            while self._started:
                if already:
                    self._scheduleSimulate()
                already = True
                self._inCFLoop = True
                try:
                    self._runner()
                finally:
                    self._inCFLoop = False
        finally:
            self._stopSimulating()
    _currentSimulator: object | None = None

    def _stopSimulating(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        If we have a CFRunLoopTimer registered with the CFRunLoop, invalidate\n        it and set it to None.\n        '
        if self._currentSimulator is None:
            return
        CFRunLoopTimerInvalidate(self._currentSimulator)
        self._currentSimulator = None

    def _scheduleSimulate(self, force: bool=False) -> None:
        if False:
            return 10
        '\n        Schedule a call to C{self.runUntilCurrent}.  This will cancel the\n        currently scheduled call if it is already scheduled.\n\n        @param force: Even if there are no timed calls, make sure that\n            C{runUntilCurrent} runs immediately (in a 0-seconds-from-now\n            C{CFRunLoopTimer}).  This is necessary for calls which need to\n            trigger behavior of C{runUntilCurrent} other than running timed\n            calls, such as draining the thread call queue or calling C{crash()}\n            when the appropriate flags are set.\n\n        @type force: C{bool}\n        '
        self._stopSimulating()
        if not self._started:
            return
        timeout = 0.0 if force else self.timeout()
        if timeout is None:
            return
        fireDate = CFAbsoluteTimeGetCurrent() + timeout

        def simulate(cftimer, extra):
            if False:
                print('Hello World!')
            self._currentSimulator = None
            self.runUntilCurrent()
            self._scheduleSimulate()
        c = self._currentSimulator = CFRunLoopTimerCreate(kCFAllocatorDefault, fireDate, 0, 0, 0, simulate, None)
        CFRunLoopAddTimer(self._cfrunloop, c, kCFRunLoopCommonModes)

    def callLater(self, _seconds, _f, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement L{IReactorTime.callLater}.\n        '
        delayedCall = PosixReactorBase.callLater(self, _seconds, _f, *args, **kw)
        self._scheduleSimulate()
        return delayedCall

    def stop(self):
        if False:
            return 10
        '\n        Implement L{IReactorCore.stop}.\n        '
        PosixReactorBase.stop(self)
        self._scheduleSimulate(True)

    def crash(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IReactorCore.crash}\n        '
        PosixReactorBase.crash(self)
        if not self._inCFLoop:
            return
        CFRunLoopStop(self._cfrunloop)

    def iterate(self, delay=0):
        if False:
            while True:
                i = 10
        '\n        Emulate the behavior of C{iterate()} for things that want to call it,\n        by letting the loop run for a little while and then scheduling a timed\n        call to exit it.\n        '
        self._started = True
        self.callLater(0, self.crash)
        self.mainLoop()

def install(runLoop=None, runner=None):
    if False:
        while True:
            i = 10
    '\n    Configure the twisted mainloop to be run inside CFRunLoop.\n\n    @param runLoop: the run loop to use.\n\n    @param runner: the function to call in order to actually invoke the main\n        loop.  This will default to C{CFRunLoopRun} if not specified.  However,\n        this is not an appropriate choice for GUI applications, as you need to\n        run NSApplicationMain (or something like it).  For example, to run the\n        Twisted mainloop in a PyObjC application, your C{main.py} should look\n        something like this::\n\n            from PyObjCTools import AppHelper\n            from twisted.internet.cfreactor import install\n            install(runner=AppHelper.runEventLoop)\n            # initialize your application\n            reactor.run()\n\n    @return: The installed reactor.\n\n    @rtype: C{CFReactor}\n    '
    reactor = CFReactor(runLoop=runLoop, runner=runner)
    from twisted.internet.main import installReactor
    installReactor(reactor)
    return reactor