"""
This module provides base support for Twisted to interact with the glib/gtk
mainloops.

The classes in this module should not be used directly, but rather you should
import gireactor or gtk3reactor for GObject Introspection based applications,
or glib2reactor or gtk2reactor for applications using legacy static bindings.
"""
import sys
from typing import Any, Callable, Dict, Set
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor, IWriteDescriptor
from twisted.python import log
from twisted.python.monkey import MonkeyPatcher
from ._signals import _UnixWaker

def ensureNotImported(moduleNames, errorMessage, preventImports=[]):
    if False:
        return 10
    "\n    Check whether the given modules were imported, and if requested, ensure\n    they will not be importable in the future.\n\n    @param moduleNames: A list of module names we make sure aren't imported.\n    @type moduleNames: C{list} of C{str}\n\n    @param preventImports: A list of module name whose future imports should\n        be prevented.\n    @type preventImports: C{list} of C{str}\n\n    @param errorMessage: Message to use when raising an C{ImportError}.\n    @type errorMessage: C{str}\n\n    @raise ImportError: with given error message if a given module name\n        has already been imported.\n    "
    for name in moduleNames:
        if sys.modules.get(name) is not None:
            raise ImportError(errorMessage)
    for name in preventImports:
        sys.modules[name] = None

class GlibWaker(_UnixWaker):
    """
    Run scheduled events after waking up.
    """

    def __init__(self, reactor):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.reactor = reactor

    def doRead(self) -> None:
        if False:
            i = 10
            return i + 15
        super().doRead()
        self.reactor._simulate()

def _signalGlue():
    if False:
        i = 10
        return i + 15
    "\n    Integrate glib's wakeup file descriptor usage and our own.\n\n    Python supports only one wakeup file descriptor at a time and both Twisted\n    and glib want to use it.\n\n    This is a context manager that can be wrapped around the whole glib\n    reactor main loop which makes our signal handling work with glib's signal\n    handling.\n    "
    from gi import _ossighelper as signalGlue
    patcher = MonkeyPatcher()
    patcher.addPatch(signalGlue, '_wakeup_fd_is_active', True)
    return patcher

def _loopQuitter(idleAdd: Callable[[Callable[[], None]], None], loopQuit: Callable[[], None]) -> Callable[[], None]:
    if False:
        print('Hello World!')
    '\n    Combine the C{glib.idle_add} and C{glib.MainLoop.quit} functions into a\n    function suitable for crashing the reactor.\n    '
    return lambda : idleAdd(loopQuit)

@implementer(IReactorFDSet)
class GlibReactorBase(posixbase.PosixReactorBase, posixbase._PollLikeMixin):
    """
    Base class for GObject event loop reactors.

    Notification for I/O events (reads and writes on file descriptors) is done
    by the gobject-based event loop. File descriptors are registered with
    gobject with the appropriate flags for read/write/disconnect notification.

    Time-based events, the results of C{callLater} and C{callFromThread}, are
    handled differently. Rather than registering each event with gobject, a
    single gobject timeout is registered for the earliest scheduled event, the
    output of C{reactor.timeout()}. For example, if there are timeouts in 1, 2
    and 3.4 seconds, a single timeout is registered for 1 second in the
    future. When this timeout is hit, C{_simulate} is called, which calls the
    appropriate Twisted-level handlers, and a new timeout is added to gobject
    by the C{_reschedule} method.

    To handle C{callFromThread} events, we use a custom waker that calls
    C{_simulate} whenever it wakes up.

    @ivar _sources: A dictionary mapping L{FileDescriptor} instances to
        GSource handles.

    @ivar _reads: A set of L{FileDescriptor} instances currently monitored for
        reading.

    @ivar _writes: A set of L{FileDescriptor} instances currently monitored for
        writing.

    @ivar _simtag: A GSource handle for the next L{simulate} call.
    """

    def _wakerFactory(self) -> GlibWaker:
        if False:
            print('Hello World!')
        return GlibWaker(self)

    def __init__(self, glib_module: Any, gtk_module: Any, useGtk: bool=False) -> None:
        if False:
            while True:
                i = 10
        self._simtag = None
        self._reads: Set[IReadDescriptor] = set()
        self._writes: Set[IWriteDescriptor] = set()
        self._sources: Dict[FileDescriptor, int] = {}
        self._glib = glib_module
        self._POLL_DISCONNECTED = glib_module.IOCondition.HUP | glib_module.IOCondition.ERR | glib_module.IOCondition.NVAL
        self._POLL_IN = glib_module.IOCondition.IN
        self._POLL_OUT = glib_module.IOCondition.OUT
        self.INFLAGS = self._POLL_IN | self._POLL_DISCONNECTED
        self.OUTFLAGS = self._POLL_OUT | self._POLL_DISCONNECTED
        super().__init__()
        self._source_remove = self._glib.source_remove
        self._timeout_add = self._glib.timeout_add
        self.context = self._glib.main_context_default()
        self._pending = self.context.pending
        self._iteration = self.context.iteration
        self.loop = self._glib.MainLoop()
        self._crash = _loopQuitter(self._glib.idle_add, self.loop.quit)
        self._run = self.loop.run

    def _reallyStartRunning(self):
        if False:
            print('Hello World!')
        "\n        Make sure the reactor's signal handlers are installed despite any\n        outside interference.\n        "
        super()._reallyStartRunning()

        def reinitSignals():
            if False:
                for i in range(10):
                    print('nop')
            self._signals.uninstall()
            self._signals.install()
        self.callLater(0, reinitSignals)

    def input_add(self, source, condition, callback):
        if False:
            while True:
                i = 10
        if hasattr(source, 'fileno'):

            def wrapper(ignored, condition):
                if False:
                    return 10
                return callback(source, condition)
            fileno = source.fileno()
        else:
            fileno = source
            wrapper = callback
        return self._glib.io_add_watch(fileno, self._glib.PRIORITY_DEFAULT_IDLE, condition, wrapper)

    def _ioEventCallback(self, source, condition):
        if False:
            print('Hello World!')
        '\n        Called by event loop when an I/O event occurs.\n        '
        log.callWithLogger(source, self._doReadOrWrite, source, source, condition)
        return True

    def _add(self, source, primary, other, primaryFlag, otherFlag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add the given L{FileDescriptor} for monitoring either for reading or\n        writing. If the file is already monitored for the other operation, we\n        delete the previous registration and re-register it for both reading\n        and writing.\n        '
        if source in primary:
            return
        flags = primaryFlag
        if source in other:
            self._source_remove(self._sources[source])
            flags |= otherFlag
        self._sources[source] = self.input_add(source, flags, self._ioEventCallback)
        primary.add(source)

    def addReader(self, reader):
        if False:
            while True:
                i = 10
        '\n        Add a L{FileDescriptor} for monitoring of data available to read.\n        '
        self._add(reader, self._reads, self._writes, self.INFLAGS, self.OUTFLAGS)

    def addWriter(self, writer):
        if False:
            while True:
                i = 10
        '\n        Add a L{FileDescriptor} for monitoring ability to write data.\n        '
        self._add(writer, self._writes, self._reads, self.OUTFLAGS, self.INFLAGS)

    def getReaders(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the list of current L{FileDescriptor} monitored for reading.\n        '
        return list(self._reads)

    def getWriters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve the list of current L{FileDescriptor} monitored for writing.\n        '
        return list(self._writes)

    def removeAll(self):
        if False:
            i = 10
            return i + 15
        '\n        Remove monitoring for all registered L{FileDescriptor}s.\n        '
        return self._removeAll(self._reads, self._writes)

    def _remove(self, source, primary, other, flags):
        if False:
            return 10
        "\n        Remove monitoring the given L{FileDescriptor} for either reading or\n        writing. If it's still monitored for the other operation, we\n        re-register the L{FileDescriptor} for only that operation.\n        "
        if source not in primary:
            return
        self._source_remove(self._sources[source])
        primary.remove(source)
        if source in other:
            self._sources[source] = self.input_add(source, flags, self._ioEventCallback)
        else:
            self._sources.pop(source)

    def removeReader(self, reader):
        if False:
            return 10
        '\n        Stop monitoring the given L{FileDescriptor} for reading.\n        '
        self._remove(reader, self._reads, self._writes, self.OUTFLAGS)

    def removeWriter(self, writer):
        if False:
            while True:
                i = 10
        '\n        Stop monitoring the given L{FileDescriptor} for writing.\n        '
        self._remove(writer, self._writes, self._reads, self.INFLAGS)

    def iterate(self, delay=0):
        if False:
            print('Hello World!')
        "\n        One iteration of the event loop, for trial's use.\n\n        This is not used for actual reactor runs.\n        "
        self.runUntilCurrent()
        while self._pending():
            self._iteration(0)

    def crash(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Crash the reactor.\n        '
        posixbase.PosixReactorBase.crash(self)
        self._crash()

    def stop(self):
        if False:
            return 10
        '\n        Stop the reactor.\n        '
        posixbase.PosixReactorBase.stop(self)
        self.wakeUp()

    def run(self, installSignalHandlers=True):
        if False:
            return 10
        '\n        Run the reactor.\n        '
        with _signalGlue():
            self.callWhenRunning(self._reschedule)
            self.startRunning(installSignalHandlers=installSignalHandlers)
            if self._started:
                self._run()

    def callLater(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Schedule a C{DelayedCall}.\n        '
        result = posixbase.PosixReactorBase.callLater(self, *args, **kwargs)
        self._reschedule()
        return result

    def _reschedule(self):
        if False:
            i = 10
            return i + 15
        '\n        Schedule a glib timeout for C{_simulate}.\n        '
        if self._simtag is not None:
            self._source_remove(self._simtag)
            self._simtag = None
        timeout = self.timeout()
        if timeout is not None:
            self._simtag = self._timeout_add(int(timeout * 1000), self._simulate, priority=self._glib.PRIORITY_DEFAULT_IDLE)

    def _simulate(self):
        if False:
            print('Hello World!')
        '\n        Run timers, and then reschedule glib timeout for next scheduled event.\n        '
        self.runUntilCurrent()
        self._reschedule()