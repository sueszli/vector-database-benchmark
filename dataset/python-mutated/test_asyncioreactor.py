"""
Tests for L{twisted.internet.asyncioreactor}.
"""
import gc
import sys
from asyncio import AbstractEventLoop, AbstractEventLoopPolicy, DefaultEventLoopPolicy, Future, SelectorEventLoop, get_event_loop, get_event_loop_policy, set_event_loop, set_event_loop_policy
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
hasWindowsProactorEventLoopPolicy = False
hasWindowsSelectorEventLoopPolicy = False
try:
    if sys.platform.startswith('win32'):
        from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
        hasWindowsProactorEventLoopPolicy = True
        hasWindowsSelectorEventLoopPolicy = True
except ImportError:
    pass
_defaultEventLoop = DefaultEventLoopPolicy().new_event_loop()
_defaultEventLoopIsSelector = isinstance(_defaultEventLoop, SelectorEventLoop)
_defaultEventLoop.close()

class AsyncioSelectorReactorTests(ReactorBuilder, SynchronousTestCase):
    """
    L{AsyncioSelectorReactor} tests.
    """

    def assertReactorWorksWithAsyncioFuture(self, reactor):
        if False:
            return 10
        '\n        Ensure that C{reactor} has an event loop that works\n        properly with L{asyncio.Future}.\n        '
        future = Future()
        result = []

        def completed(future):
            if False:
                i = 10
                return i + 15
            result.append(future.result())
            reactor.stop()
        future.add_done_callback(completed)
        future.set_result(True)
        self.assertEqual(result, [])
        self.runReactor(reactor, timeout=1)
        self.assertEqual(result, [True])

    def newLoop(self, policy: AbstractEventLoopPolicy) -> AbstractEventLoop:
        if False:
            i = 10
            return i + 15
        '\n        Make a new asyncio loop from a policy for use with a reactor, and add\n        appropriate cleanup to restore any global state.\n        '
        existingLoop = get_event_loop()
        existingPolicy = get_event_loop_policy()
        result = policy.new_event_loop()

        @self.addCleanup
        def cleanUp():
            if False:
                print('Hello World!')
            result.close()
            set_event_loop(existingLoop)
            set_event_loop_policy(existingPolicy)
        return result

    @skipIf(not _defaultEventLoopIsSelector, 'default event loop: {}\nis not of type SelectorEventLoop on Python {}.{} ({})'.format(type(_defaultEventLoop), sys.version_info.major, sys.version_info.minor, platform.getType()))
    def test_defaultSelectorEventLoopFromGlobalPolicy(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{AsyncioSelectorReactor} wraps the global policy's event loop\n        by default.  This ensures that L{asyncio.Future}s and\n        coroutines created by library code that uses\n        L{asyncio.get_event_loop} are bound to the same loop.\n        "
        reactor = AsyncioSelectorReactor()
        self.assertReactorWorksWithAsyncioFuture(reactor)

    @skipIf(not _defaultEventLoopIsSelector, 'default event loop: {}\nis not of type SelectorEventLoop on Python {}.{} ({})'.format(type(_defaultEventLoop), sys.version_info.major, sys.version_info.minor, platform.getType()))
    def test_newSelectorEventLoopFromDefaultEventLoopPolicy(self):
        if False:
            return 10
        '\n        If we use the L{asyncio.DefaultLoopPolicy} to create a new event loop,\n        and then pass that event loop to a new L{AsyncioSelectorReactor},\n        this reactor should work properly with L{asyncio.Future}.\n        '
        event_loop = self.newLoop(DefaultEventLoopPolicy())
        reactor = AsyncioSelectorReactor(event_loop)
        set_event_loop(event_loop)
        self.assertReactorWorksWithAsyncioFuture(reactor)

    @skipIf(_defaultEventLoopIsSelector, 'default event loop: {}\nis of type SelectorEventLoop on Python {}.{} ({})'.format(type(_defaultEventLoop), sys.version_info.major, sys.version_info.minor, platform.getType()))
    def test_defaultNotASelectorEventLoopFromGlobalPolicy(self):
        if False:
            return 10
        '\n        On Windows Python 3.5 to 3.7, L{get_event_loop()} returns a\n        L{WindowsSelectorEventLoop} by default.\n        On Windows Python 3.8+, L{get_event_loop()} returns a\n        L{WindowsProactorEventLoop} by default.\n        L{AsyncioSelectorReactor} should raise a\n        L{TypeError} if the default event loop is not a\n        L{WindowsSelectorEventLoop}.\n        '
        self.assertRaises(TypeError, AsyncioSelectorReactor)

    @skipIf(not hasWindowsProactorEventLoopPolicy, 'WindowsProactorEventLoop not available')
    def test_WindowsProactorEventLoop(self):
        if False:
            return 10
        '\n        L{AsyncioSelectorReactor} will raise a L{TypeError}\n        if instantiated with a L{asyncio.WindowsProactorEventLoop}\n        '
        event_loop = self.newLoop(WindowsProactorEventLoopPolicy())
        self.assertRaises(TypeError, AsyncioSelectorReactor, event_loop)

    @skipIf(not hasWindowsSelectorEventLoopPolicy, 'WindowsSelectorEventLoop only on Windows')
    def test_WindowsSelectorEventLoop(self):
        if False:
            while True:
                i = 10
        '\n        L{WindowsSelectorEventLoop} works with L{AsyncioSelectorReactor}\n        '
        event_loop = self.newLoop(WindowsSelectorEventLoopPolicy())
        reactor = AsyncioSelectorReactor(event_loop)
        set_event_loop(event_loop)
        self.assertReactorWorksWithAsyncioFuture(reactor)

    @skipIf(not hasWindowsProactorEventLoopPolicy, 'WindowsProactorEventLoopPolicy only on Windows')
    def test_WindowsProactorEventLoopPolicy(self):
        if False:
            print('Hello World!')
        '\n        L{AsyncioSelectorReactor} will raise a L{TypeError}\n        if L{asyncio.WindowsProactorEventLoopPolicy} is default.\n        '
        set_event_loop_policy(WindowsProactorEventLoopPolicy())
        self.addCleanup(lambda : set_event_loop_policy(None))
        with self.assertRaises(TypeError):
            AsyncioSelectorReactor()

    @skipIf(not hasWindowsSelectorEventLoopPolicy, 'WindowsSelectorEventLoopPolicy only on Windows')
    def test_WindowsSelectorEventLoopPolicy(self):
        if False:
            while True:
                i = 10
        '\n        L{AsyncioSelectorReactor} will work if\n        if L{asyncio.WindowsSelectorEventLoopPolicy} is default.\n        '
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        self.addCleanup(lambda : set_event_loop_policy(None))
        reactor = AsyncioSelectorReactor()
        self.assertReactorWorksWithAsyncioFuture(reactor)

    def test_seconds(self):
        if False:
            while True:
                i = 10
        'L{seconds} should return a plausible epoch time.'
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(WindowsSelectorEventLoopPolicy())
            self.addCleanup(lambda : set_event_loop_policy(None))
        reactor = AsyncioSelectorReactor()
        result = reactor.seconds()
        self.assertGreater(result, 1577836800)
        self.assertLess(result, 4733510400)

    def test_delayedCallResetToLater(self):
        if False:
            print('Hello World!')
        '\n        L{DelayedCall.reset()} properly reschedules timer to later time\n        '
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(WindowsSelectorEventLoopPolicy())
            self.addCleanup(lambda : set_event_loop_policy(None))
        reactor = AsyncioSelectorReactor()
        timer_called_at = [None]

        def on_timer():
            if False:
                return 10
            timer_called_at[0] = reactor.seconds()
        start_time = reactor.seconds()
        dc = reactor.callLater(0, on_timer)
        dc.reset(0.5)
        reactor.callLater(1, reactor.stop)
        reactor.run()
        self.assertIsNotNone(timer_called_at[0])
        self.assertGreater(timer_called_at[0] - start_time, 0.4)

    def test_delayedCallResetToEarlier(self):
        if False:
            i = 10
            return i + 15
        '\n        L{DelayedCall.reset()} properly reschedules timer to earlier time\n        '
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        reactor = AsyncioSelectorReactor()
        timer_called_at = [None]

        def on_timer():
            if False:
                i = 10
                return i + 15
            timer_called_at[0] = reactor.seconds()
        start_time = reactor.seconds()
        dc = reactor.callLater(0.5, on_timer)
        dc.reset(0)
        reactor.callLater(1, reactor.stop)
        import io
        from contextlib import redirect_stderr
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            reactor.run()
        self.assertEqual(stderr.getvalue(), '')
        self.assertIsNotNone(timer_called_at[0])
        self.assertLess(timer_called_at[0] - start_time, 0.4)
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(None)

    def test_noCycleReferencesInCallLater(self):
        if False:
            while True:
                i = 10
        "\n        L{AsyncioSelectorReactor.callLater()} doesn't leave cyclic references\n        "
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            objects_before = len(gc.get_objects())
            timer_count = 1000
            reactor = AsyncioSelectorReactor()
            for _ in range(timer_count):
                reactor.callLater(0, lambda : None)
            reactor.runUntilCurrent()
            objects_after = len(gc.get_objects())
            self.assertLess((objects_after - objects_before) / timer_count, 1)
        finally:
            if gc_was_enabled:
                gc.enable()
        if hasWindowsSelectorEventLoopPolicy:
            set_event_loop_policy(None)