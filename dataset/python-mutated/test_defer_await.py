"""
Tests for C{await} support in Deferreds.
"""
import types
from typing_extensions import NoReturn
from twisted.internet.defer import Deferred, ensureDeferred, fail, maybeDeferred, succeed
from twisted.internet.task import Clock
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase

class SampleException(Exception):
    """
    A specific sample exception for testing.
    """

class AwaitTests(TestCase):
    """
    Tests for using Deferreds in conjunction with PEP-492.
    """

    def test_awaitReturnsIterable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        C{Deferred.__await__} returns an iterable.\n        '
        d: Deferred[None] = Deferred()
        awaitedDeferred = d.__await__()
        self.assertEqual(awaitedDeferred, iter(awaitedDeferred))

    def test_deferredFromCoroutine(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Deferred.fromCoroutine} will turn a coroutine into a L{Deferred}.\n        '

        async def run() -> str:
            d = succeed('bar')
            await d
            res = await run2()
            return res

        async def run2() -> str:
            d = succeed('foo')
            res = await d
            return res
        r = run()
        self.assertIsInstance(r, types.CoroutineType)
        d = Deferred.fromCoroutine(r)
        self.assertIsInstance(d, Deferred)
        res = self.successResultOf(d)
        self.assertEqual(res, 'foo')

    def test_basic(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{Deferred.fromCoroutine} allows a function to C{await} on a\n        L{Deferred}.\n        '

        async def run() -> str:
            d = succeed('foo')
            res = await d
            return res
        d = Deferred.fromCoroutine(run())
        res = self.successResultOf(d)
        self.assertEqual(res, 'foo')

    def test_basicEnsureDeferred(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{ensureDeferred} allows a function to C{await} on a L{Deferred}.\n        '

        async def run() -> str:
            d = succeed('foo')
            res = await d
            return res
        d = ensureDeferred(run())
        res = self.successResultOf(d)
        self.assertEqual(res, 'foo')

    def test_exception(self) -> None:
        if False:
            while True:
                i = 10
        '\n        An exception in a coroutine scheduled with L{Deferred.fromCoroutine}\n        will cause the returned L{Deferred} to fire with a failure.\n        '

        async def run() -> NoReturn:
            d = succeed('foo')
            await d
            raise ValueError('Oh no!')
        d = Deferred.fromCoroutine(run())
        res = self.failureResultOf(d)
        self.assertEqual(type(res.value), ValueError)
        self.assertEqual(res.value.args, ('Oh no!',))

    def test_synchronousDeferredFailureTraceback(self) -> None:
        if False:
            print('Hello World!')
        '\n        When a Deferred is awaited upon that has already failed with a Failure\n        that has a traceback, both the place that the synchronous traceback\n        comes from and the awaiting line are shown in the traceback.\n        '

        def raises() -> None:
            if False:
                for i in range(10):
                    print('nop')
            raise SampleException()
        it = maybeDeferred(raises)

        async def doomed() -> None:
            return await it
        failure = self.failureResultOf(Deferred.fromCoroutine(doomed()))
        self.assertIn(', in doomed\n', failure.getTraceback())
        self.assertIn(', in raises\n', failure.getTraceback())

    def test_asyncDeferredFailureTraceback(self) -> None:
        if False:
            return 10
        '\n        When a Deferred is awaited upon that later fails with a Failure that\n        has a traceback, both the place that the synchronous traceback comes\n        from and the awaiting line are shown in the traceback.\n        '

        def returnsFailure() -> Failure:
            if False:
                while True:
                    i = 10
            try:
                raise SampleException()
            except SampleException:
                return Failure()
        it: Deferred[None] = Deferred()

        async def doomed() -> None:
            return await it
        started = Deferred.fromCoroutine(doomed())
        self.assertNoResult(started)
        it.errback(returnsFailure())
        failure = self.failureResultOf(started)
        self.assertIn(', in doomed\n', failure.getTraceback())
        self.assertIn(', in returnsFailure\n', failure.getTraceback())

    def test_twoDeep(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        A coroutine scheduled with L{Deferred.fromCoroutine} that awaits a\n        L{Deferred} suspends its execution until the inner L{Deferred} fires.\n        '
        reactor = Clock()
        sections = []

        async def runone() -> str:
            sections.append(2)
            d: Deferred[int] = Deferred()
            reactor.callLater(1, d.callback, 2)
            await d
            sections.append(3)
            return 'Yay!'

        async def run() -> str:
            sections.append(1)
            result = await runone()
            sections.append(4)
            d: Deferred[int] = Deferred()
            reactor.callLater(1, d.callback, 1)
            await d
            sections.append(5)
            return result
        d = Deferred.fromCoroutine(run())
        reactor.advance(0.9)
        self.assertEqual(sections, [1, 2])
        reactor.advance(0.1)
        self.assertEqual(sections, [1, 2, 3, 4])
        reactor.advance(0.9)
        self.assertEqual(sections, [1, 2, 3, 4])
        reactor.advance(0.1)
        self.assertEqual(sections, [1, 2, 3, 4, 5])
        res = self.successResultOf(d)
        self.assertEqual(res, 'Yay!')

    def test_reraise(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Awaiting an already failed Deferred will raise the exception.\n        '

        async def test() -> int:
            try:
                await fail(ValueError('Boom'))
            except ValueError as e:
                self.assertEqual(e.args, ('Boom',))
                return 1
            return 0
        res = self.successResultOf(Deferred.fromCoroutine(test()))
        self.assertEqual(res, 1)

    def test_chained(self) -> None:
        if False:
            print('Hello World!')
        '\n        Awaiting a paused & chained Deferred will give the result when it has\n        one.\n        '
        reactor = Clock()

        async def test() -> None:
            d: Deferred[None] = Deferred()
            d2: Deferred[None] = Deferred()
            d.addCallback(lambda ignored: d2)
            d.callback(None)
            reactor.callLater(0, d2.callback, 'bye')
            return await d
        d = Deferred.fromCoroutine(test())
        reactor.advance(0.1)
        res = self.successResultOf(d)
        self.assertEqual(res, 'bye')