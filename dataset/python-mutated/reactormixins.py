"""
Utilities for unit testing reactor implementations.

The main feature of this module is L{ReactorBuilder}, a base class for use when
writing interface/blackbox tests for reactor implementations.  Test case classes
for reactor features should subclass L{ReactorBuilder} instead of
L{SynchronousTestCase}.  All of the features of L{SynchronousTestCase} will be
available.  Additionally, the tests will automatically be applied to all
available reactor implementations.
"""
__all__ = ['TestTimeoutError', 'ReactorBuilder', 'needsRunningReactor']
import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
if TYPE_CHECKING:
    from twisted.internet import asyncioreactor
try:
    from twisted.internet import process as _process
except ImportError:
    process = None
else:
    process = _process

class TestTimeoutError(Exception):
    """
    The reactor was still running after the timeout period elapsed in
    L{ReactorBuilder.runReactor}.
    """

def needsRunningReactor(reactor, thunk):
    if False:
        return 10
    "\n    Various functions within these tests need an already-running reactor at\n    some point.  They need to stop the reactor when the test has completed, and\n    that means calling reactor.stop().  However, reactor.stop() raises an\n    exception if the reactor isn't already running, so if the L{Deferred} that\n    a particular API under test returns fires synchronously (as especially an\n    endpoint's C{connect()} method may do, if the connect is to a local\n    interface address) then the test won't be able to stop the reactor being\n    tested and finish.  So this calls C{thunk} only once C{reactor} is running.\n\n    (This is just an alias for\n    L{twisted.internet.interfaces.IReactorCore.callWhenRunning} on the given\n    reactor parameter, in order to centrally reference the above paragraph and\n    repeating it everywhere as a comment.)\n\n    @param reactor: the L{twisted.internet.interfaces.IReactorCore} under test\n\n    @param thunk: a 0-argument callable, which eventually finishes the test in\n        question, probably in a L{Deferred} callback.\n    "
    reactor.callWhenRunning(thunk)

def stopOnError(case, reactor, publisher=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the reactor as soon as any error is logged on the given publisher.\n\n    This is beneficial for tests which will wait for a L{Deferred} to fire\n    before completing (by passing or failing).  Certain implementation bugs may\n    prevent the L{Deferred} from firing with any result at all (consider a\n    protocol's {dataReceived} method that raises an exception: this exception\n    is logged but it won't ever cause a L{Deferred} to fire).  In that case the\n    test would have to complete by timing out which is a much less desirable\n    outcome than completing as soon as the unexpected error is encountered.\n\n    @param case: A L{SynchronousTestCase} to use to clean up the necessary log\n        observer when the test is over.\n    @param reactor: The reactor to stop.\n    @param publisher: A L{LogPublisher} to watch for errors.  If L{None}, the\n        global log publisher will be watched.\n    "
    if publisher is None:
        from twisted.python import log as publisher
    running = [None]

    def stopIfError(event):
        if False:
            return 10
        if running and event.get('isError'):
            running.pop()
            reactor.stop()
    publisher.addObserver(stopIfError)
    case.addCleanup(publisher.removeObserver, stopIfError)

class ReactorBuilder:
    """
    L{SynchronousTestCase} mixin which provides a reactor-creation API.  This
    mixin defines C{setUp} and C{tearDown}, so mix it in before
    L{SynchronousTestCase} or call its methods from the overridden ones in the
    subclass.

    @cvar skippedReactors: A dict mapping FQPN strings of reactors for
        which the tests defined by this class will be skipped to strings
        giving the skip message.
    @cvar requiredInterfaces: A C{list} of interfaces which the reactor must
        provide or these tests will be skipped.  The default, L{None}, means
        that no interfaces are required.
    @ivar reactorFactory: A no-argument callable which returns the reactor to
        use for testing.
    @ivar originalHandler: The SIGCHLD handler which was installed when setUp
        ran and which will be re-installed when tearDown runs.
    @ivar _reactors: A list of FQPN strings giving the reactors for which
        L{SynchronousTestCase}s will be created.
    """
    _reactors = ['twisted.internet.selectreactor.SelectReactor']
    if platform.isWindows():
        _reactors.extend(['twisted.internet.gireactor.PortableGIReactor', 'twisted.internet.win32eventreactor.Win32Reactor', 'twisted.internet.iocpreactor.reactor.IOCPReactor'])
    else:
        _reactors.extend(['twisted.internet.gireactor.GIReactor'])
        _reactors.append('twisted.internet.test.reactormixins.AsyncioSelectorReactor')
        if platform.isMacOSX():
            _reactors.append('twisted.internet.cfreactor.CFReactor')
        else:
            _reactors.extend(['twisted.internet.pollreactor.PollReactor', 'twisted.internet.epollreactor.EPollReactor'])
            if not platform.isLinux():
                _reactors.extend(['twisted.internet.kqreactor.KQueueReactor'])
    reactorFactory: Optional[Callable[[], object]] = None
    originalHandler = None
    requiredInterfaces: Optional[Sequence[Type[Interface]]] = None
    skippedReactors: Dict[str, str] = {}

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Clear the SIGCHLD handler, if there is one, to ensure an environment\n        like the one which exists prior to a call to L{reactor.run}.\n        '
        if not platform.isWindows():
            self.originalHandler = signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore the original SIGCHLD handler and reap processes as long as\n        there seem to be any remaining.\n        '
        if self.originalHandler is not None:
            signal.signal(signal.SIGCHLD, self.originalHandler)
        if process is not None:
            begin = time.time()
            while process.reapProcessHandlers:
                log.msg('ReactorBuilder.tearDown reaping some processes %r' % (process.reapProcessHandlers,))
                process.reapAllProcesses()
                time.sleep(0.001)
                if time.time() - begin > 60:
                    for pid in process.reapProcessHandlers:
                        os.kill(pid, signal.SIGKILL)
                    raise Exception('Timeout waiting for child processes to exit: %r' % (process.reapProcessHandlers,))

    def _unbuildReactor(self, reactor):
        if False:
            i = 10
            return i + 15
        '\n        Clean up any resources which may have been allocated for the given\n        reactor by its creation or by a test which used it.\n        '
        reactor._uninstallHandler()
        if getattr(reactor, '_internalReaders', None) is not None:
            for reader in reactor._internalReaders:
                reactor.removeReader(reader)
                reader.connectionLost(None)
            reactor._internalReaders.clear()
        reactor.disconnectAll()
        calls = reactor.getDelayedCalls()
        for c in calls:
            c.cancel()
        from twisted.internet import reactor as globalReactor
        globalReactor.__dict__ = reactor._originalReactorDict
        globalReactor.__class__ = reactor._originalReactorClass

    def buildReactor(self):
        if False:
            while True:
                i = 10
        '\n        Create and return a reactor using C{self.reactorFactory}.\n        '
        try:
            from twisted.internet import reactor as globalReactor
            from twisted.internet.cfreactor import CFReactor
        except ImportError:
            pass
        else:
            if isinstance(globalReactor, CFReactor) and self.reactorFactory is CFReactor:
                raise SkipTest("CFReactor uses APIs which manipulate global state, so it's not safe to run its own reactor-builder tests under itself")
        try:
            assert self.reactorFactory is not None
            reactor = self.reactorFactory()
            reactor._originalReactorDict = globalReactor.__dict__
            reactor._originalReactorClass = globalReactor.__class__
            globalReactor.__dict__ = reactor.__dict__
            globalReactor.__class__ = reactor.__class__
        except BaseException:
            log.err(None, 'Failed to install reactor')
            self.flushLoggedErrors()
            raise SkipTest(Failure().getErrorMessage())
        else:
            if self.requiredInterfaces is not None:
                missing = [required for required in self.requiredInterfaces if not required.providedBy(reactor)]
                if missing:
                    self._unbuildReactor(reactor)
                    raise SkipTest('%s does not provide %s' % (fullyQualifiedName(reactor.__class__), ','.join([fullyQualifiedName(x) for x in missing])))
        self.addCleanup(self._unbuildReactor, reactor)
        return reactor

    def getTimeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine how long to run the test before considering it failed.\n\n        @return: A C{int} or C{float} giving a number of seconds.\n        '
        return acquireAttribute(self._parents, 'timeout', DEFAULT_TIMEOUT_DURATION)

    def runReactor(self, reactor, timeout=None):
        if False:
            print('Hello World!')
        '\n        Run the reactor for at most the given amount of time.\n\n        @param reactor: The reactor to run.\n\n        @type timeout: C{int} or C{float}\n        @param timeout: The maximum amount of time, specified in seconds, to\n            allow the reactor to run.  If the reactor is still running after\n            this much time has elapsed, it will be stopped and an exception\n            raised.  If L{None}, the default test method timeout imposed by\n            Trial will be used.  This depends on the L{IReactorTime}\n            implementation of C{reactor} for correct operation.\n\n        @raise TestTimeoutError: If the reactor is still running after\n            C{timeout} seconds.\n        '
        if timeout is None:
            timeout = self.getTimeout()
        timedOut = []

        def stop():
            if False:
                return 10
            timedOut.append(None)
            reactor.stop()
        timedOutCall = reactor.callLater(timeout, stop)
        reactor.run()
        if timedOut:
            raise TestTimeoutError(f'reactor still running after {timeout} seconds')
        else:
            timedOutCall.cancel()

    @classmethod
    def makeTestCaseClasses(cls: Type['ReactorBuilder']) -> Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]]:
        if False:
            i = 10
            return i + 15
        '\n        Create a L{SynchronousTestCase} subclass which mixes in C{cls} for each\n        known reactor and return a dict mapping their names to them.\n        '
        classes: Dict[str, Union[Type['ReactorBuilder'], Type[SynchronousTestCase]]] = {}
        for reactor in cls._reactors:
            shortReactorName = reactor.split('.')[-1]
            name = (cls.__name__ + '.' + shortReactorName + 'Tests').replace('.', '_')

            class testcase(cls, SynchronousTestCase):
                __module__ = cls.__module__
                if reactor in cls.skippedReactors:
                    skip = cls.skippedReactors[reactor]
                try:
                    reactorFactory = namedAny(reactor)
                except BaseException:
                    skip = Failure().getErrorMessage()
            testcase.__name__ = name
            testcase.__qualname__ = '.'.join(cls.__qualname__.split()[0:-1] + [name])
            classes[testcase.__name__] = testcase
        return classes

def asyncioSelectorReactor(self: object) -> 'asyncioreactor.AsyncioSelectorReactor':
    if False:
        while True:
            i = 10
    "\n    Make a new asyncio reactor associated with a new event loop.\n\n    The test suite prefers this constructor because having a new event loop\n    for each reactor provides better test isolation.  The real constructor\n    prefers to re-use (or create) a global loop because of how this interacts\n    with other asyncio-based libraries and applications (though maybe it\n    shouldn't).\n\n    @param self: The L{ReactorBuilder} subclass this is being called on.  We\n        don't use this parameter but we get called with it anyway.\n    "
    from asyncio import get_event_loop, new_event_loop, set_event_loop
    from twisted.internet import asyncioreactor
    asTestCase = cast(SynchronousTestCase, self)
    originalLoop = get_event_loop()
    loop = new_event_loop()
    set_event_loop(loop)

    @asTestCase.addCleanup
    def cleanUp():
        if False:
            print('Hello World!')
        loop.close()
        set_event_loop(originalLoop)
    return asyncioreactor.AsyncioSelectorReactor(loop)
AsyncioSelectorReactor = asyncioSelectorReactor