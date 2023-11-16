from typing import Callable, Generator, cast
import twisted.python.failure
from twisted.internet import defer, reactor as _reactor
from synapse.logging.context import SENTINEL_CONTEXT, LoggingContext, PreserveLoggingContext, current_context, make_deferred_yieldable, nested_logging_context, run_in_background
from synapse.types import ISynapseReactor
from synapse.util import Clock
from .. import unittest
reactor = cast(ISynapseReactor, _reactor)

class LoggingContextTestCase(unittest.TestCase):

    def _check_test_key(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        context = current_context()
        assert isinstance(context, LoggingContext)
        self.assertEqual(context.name, value)

    def test_with_context(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with LoggingContext('test'):
            self._check_test_key('test')

    @defer.inlineCallbacks
    def test_sleep(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            i = 10
            return i + 15
        clock = Clock(reactor)

        @defer.inlineCallbacks
        def competing_callback() -> Generator['defer.Deferred[object]', object, None]:
            if False:
                for i in range(10):
                    print('nop')
            with LoggingContext('competing'):
                yield clock.sleep(0)
                self._check_test_key('competing')
        reactor.callLater(0, competing_callback)
        with LoggingContext('one'):
            yield clock.sleep(0)
            self._check_test_key('one')

    def _test_run_in_background(self, function: Callable[[], object]) -> defer.Deferred:
        if False:
            for i in range(10):
                print('nop')
        sentinel_context = current_context()
        callback_completed = False
        with LoggingContext('one'):
            d2 = run_in_background(function)

            def cb(res: object) -> object:
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal callback_completed
                callback_completed = True
                return res
            d2.addCallback(cb)
            self._check_test_key('one')
        d2 = defer.Deferred()

        def check_logcontext() -> None:
            if False:
                return 10
            if not callback_completed:
                reactor.callLater(0.01, check_logcontext)
                return
            try:
                self.assertIs(current_context(), sentinel_context)
                d2.callback(None)
            except BaseException:
                d2.errback(twisted.python.failure.Failure())
        reactor.callLater(0.01, check_logcontext)
        return d2

    def test_run_in_background_with_blocking_fn(self) -> defer.Deferred:
        if False:
            i = 10
            return i + 15

        @defer.inlineCallbacks
        def blocking_function() -> Generator['defer.Deferred[object]', object, None]:
            if False:
                print('Hello World!')
            yield Clock(reactor).sleep(0)
        return self._test_run_in_background(blocking_function)

    def test_run_in_background_with_non_blocking_fn(self) -> defer.Deferred:
        if False:
            for i in range(10):
                print('nop')

        @defer.inlineCallbacks
        def nonblocking_function() -> Generator['defer.Deferred[object]', object, None]:
            if False:
                print('Hello World!')
            with PreserveLoggingContext():
                yield defer.succeed(None)
        return self._test_run_in_background(nonblocking_function)

    def test_run_in_background_with_chained_deferred(self) -> defer.Deferred:
        if False:
            while True:
                i = 10

        def testfunc() -> defer.Deferred:
            if False:
                print('Hello World!')
            return make_deferred_yieldable(_chained_deferred_function())
        return self._test_run_in_background(testfunc)

    def test_run_in_background_with_coroutine(self) -> defer.Deferred:
        if False:
            return 10

        async def testfunc() -> None:
            self._check_test_key('one')
            d = Clock(reactor).sleep(0)
            self.assertIs(current_context(), SENTINEL_CONTEXT)
            await d
            self._check_test_key('one')
        return self._test_run_in_background(testfunc)

    def test_run_in_background_with_nonblocking_coroutine(self) -> defer.Deferred:
        if False:
            for i in range(10):
                print('nop')

        async def testfunc() -> None:
            self._check_test_key('one')
        return self._test_run_in_background(testfunc)

    @defer.inlineCallbacks
    def test_make_deferred_yieldable(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            i = 10
            return i + 15

        def blocking_function() -> defer.Deferred:
            if False:
                print('Hello World!')
            d: defer.Deferred = defer.Deferred()
            reactor.callLater(0, d.callback, None)
            return d
        sentinel_context = current_context()
        with LoggingContext('one'):
            d1 = make_deferred_yieldable(blocking_function())
            self.assertIs(current_context(), sentinel_context)
            yield d1
            self._check_test_key('one')

    @defer.inlineCallbacks
    def test_make_deferred_yieldable_with_chained_deferreds(self) -> Generator['defer.Deferred[object]', object, None]:
        if False:
            while True:
                i = 10
        sentinel_context = current_context()
        with LoggingContext('one'):
            d1 = make_deferred_yieldable(_chained_deferred_function())
            self.assertIs(current_context(), sentinel_context)
            yield d1
            self._check_test_key('one')

    def test_nested_logging_context(self) -> None:
        if False:
            while True:
                i = 10
        with LoggingContext('foo'):
            nested_context = nested_logging_context(suffix='bar')
            self.assertEqual(nested_context.name, 'foo-bar')

def _chained_deferred_function() -> defer.Deferred:
    if False:
        i = 10
        return i + 15
    d = defer.succeed(None)

    def cb(res: object) -> defer.Deferred:
        if False:
            print('Hello World!')
        d2: defer.Deferred = defer.Deferred()
        reactor.callLater(0, d2.callback, res)
        return d2
    d.addCallback(cb)
    return d