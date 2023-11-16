from typing import NoReturn
from unittest.mock import Mock
from twisted.internet import defer
from twisted.internet.defer import Deferred
from synapse.util.caches.cached_call import CachedCall, RetryOnExceptionCachedCall
from tests.test_utils import get_awaitable_result
from tests.unittest import TestCase

class CachedCallTestCase(TestCase):

    def test_get(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Happy-path test case: makes a couple of calls and makes sure they behave\n        correctly\n        '
        d: 'Deferred[int]' = Deferred()

        async def f() -> int:
            return await d
        slow_call = Mock(side_effect=f)
        cached_call = CachedCall(slow_call)
        slow_call.assert_not_called()
        completed_results = []

        async def r() -> None:
            res = await cached_call.get()
            completed_results.append(res)
        r1 = defer.ensureDeferred(r())
        r2 = defer.ensureDeferred(r())
        self.assertNoResult(r1)
        self.assertNoResult(r2)
        slow_call.assert_called_once_with()
        d.callback(123)
        self.assertEqual(completed_results, [123, 123])
        self.successResultOf(r1)
        self.successResultOf(r2)
        slow_call.reset_mock()
        r3 = get_awaitable_result(cached_call.get())
        self.assertEqual(r3, 123)
        slow_call.assert_not_called()

    def test_fast_call(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test the behaviour when the underlying function completes immediately\n        '

        async def f() -> int:
            return 12
        fast_call = Mock(side_effect=f)
        cached_call = CachedCall(fast_call)
        fast_call.assert_not_called()
        self.assertEqual(get_awaitable_result(cached_call.get()), 12)
        self.assertEqual(get_awaitable_result(cached_call.get()), 12)
        fast_call.assert_called_once_with()

class RetryOnExceptionCachedCallTestCase(TestCase):

    def test_get(self) -> None:
        if False:
            i = 10
            return i + 15
        d: 'Deferred[int]' = Deferred()

        async def f1() -> NoReturn:
            await d
            raise ValueError('moo')
        slow_call = Mock(side_effect=f1)
        cached_call = RetryOnExceptionCachedCall(slow_call)
        slow_call.assert_not_called()
        completed_results = []

        async def r() -> None:
            try:
                await cached_call.get()
            except Exception as e1:
                completed_results.append(e1)
        r1 = defer.ensureDeferred(r())
        r2 = defer.ensureDeferred(r())
        self.assertNoResult(r1)
        self.assertNoResult(r2)
        slow_call.assert_called_once_with()
        d.callback(0)
        self.assertEqual(len(completed_results), 2)
        for e in completed_results:
            self.assertIsInstance(e, ValueError)
            self.assertEqual(e.args, ('moo',))
        d = Deferred()

        async def f2() -> int:
            return await d
        slow_call.reset_mock()
        slow_call.side_effect = f2
        r3 = defer.ensureDeferred(cached_call.get())
        r4 = defer.ensureDeferred(cached_call.get())
        self.assertNoResult(r3)
        self.assertNoResult(r4)
        slow_call.assert_called_once_with()
        d.callback(123)
        self.assertEqual(self.successResultOf(r3), 123)
        self.assertEqual(self.successResultOf(r4), 123)
        slow_call.reset_mock()
        self.assertEqual(get_awaitable_result(cached_call.get()), 123)
        slow_call.assert_not_called()