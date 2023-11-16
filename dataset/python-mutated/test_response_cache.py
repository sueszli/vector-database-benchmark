from unittest.mock import Mock
from parameterized import parameterized
from twisted.internet import defer
from synapse.util.caches.response_cache import ResponseCache, ResponseCacheContext
from tests.server import get_clock
from tests.unittest import TestCase

class ResponseCacheTestCase(TestCase):
    """
    A TestCase class for ResponseCache.

    The test-case function naming has some logic to it in it's parts, here's some notes about it:
        wait: Denotes tests that have an element of "waiting" before its wrapped result becomes available
              (Generally these just use .delayed_return instead of .instant_return in it's wrapped call.)
        expire: Denotes tests that test expiry after assured existence.
                (These have cache with a short timeout_ms=, shorter than will be tested through advancing the clock)
    """

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (self.reactor, self.clock) = get_clock()

    def with_cache(self, name: str, ms: int=0) -> ResponseCache:
        if False:
            for i in range(10):
                print('nop')
        return ResponseCache(self.clock, name, timeout_ms=ms)

    @staticmethod
    async def instant_return(o: str) -> str:
        return o

    async def delayed_return(self, o: str) -> str:
        await self.clock.sleep(1)
        return o

    def test_cache_hit(self) -> None:
        if False:
            while True:
                i = 10
        cache = self.with_cache('keeping_cache', ms=9001)
        expected_result = 'howdy'
        wrap_d = defer.ensureDeferred(cache.wrap(0, self.instant_return, expected_result))
        self.assertEqual(expected_result, self.successResultOf(wrap_d), 'initial wrap result should be the same')
        unexpected = Mock(spec=())
        wrap2_d = defer.ensureDeferred(cache.wrap(0, unexpected))
        unexpected.assert_not_called()
        self.assertEqual(expected_result, self.successResultOf(wrap2_d), 'cache should still have the result')

    def test_cache_miss(self) -> None:
        if False:
            while True:
                i = 10
        cache = self.with_cache('trashing_cache', ms=0)
        expected_result = 'howdy'
        wrap_d = defer.ensureDeferred(cache.wrap(0, self.instant_return, expected_result))
        self.assertEqual(expected_result, self.successResultOf(wrap_d), 'initial wrap result should be the same')
        self.assertCountEqual([], cache.keys(), 'cache should not have the result now')

    def test_cache_expire(self) -> None:
        if False:
            while True:
                i = 10
        cache = self.with_cache('short_cache', ms=1000)
        expected_result = 'howdy'
        wrap_d = defer.ensureDeferred(cache.wrap(0, self.instant_return, expected_result))
        self.assertEqual(expected_result, self.successResultOf(wrap_d))
        unexpected = Mock(spec=())
        wrap2_d = defer.ensureDeferred(cache.wrap(0, unexpected))
        unexpected.assert_not_called()
        self.assertEqual(expected_result, self.successResultOf(wrap2_d), 'cache should still have the result')
        self.reactor.pump((2,))
        self.assertCountEqual([], cache.keys(), 'cache should not have the result now')

    def test_cache_wait_hit(self) -> None:
        if False:
            return 10
        cache = self.with_cache('neutral_cache')
        expected_result = 'howdy'
        wrap_d = defer.ensureDeferred(cache.wrap(0, self.delayed_return, expected_result))
        self.assertNoResult(wrap_d)
        self.reactor.pump((2,))
        self.assertEqual(expected_result, self.successResultOf(wrap_d))

    def test_cache_wait_expire(self) -> None:
        if False:
            i = 10
            return i + 15
        cache = self.with_cache('medium_cache', ms=3000)
        expected_result = 'howdy'
        wrap_d = defer.ensureDeferred(cache.wrap(0, self.delayed_return, expected_result))
        self.assertNoResult(wrap_d)
        self.reactor.pump((1, 1))
        self.assertEqual(expected_result, self.successResultOf(wrap_d))
        unexpected = Mock(spec=())
        wrap2_d = defer.ensureDeferred(cache.wrap(0, unexpected))
        unexpected.assert_not_called()
        self.assertEqual(expected_result, self.successResultOf(wrap2_d), 'cache should still have the result')
        self.reactor.pump((2,))
        self.assertCountEqual([], cache.keys(), 'cache should not have the result now')

    @parameterized.expand([(True,), (False,)])
    def test_cache_context_nocache(self, should_cache: bool) -> None:
        if False:
            print('Hello World!')
        'If the callback clears the should_cache bit, the result should not be cached'
        cache = self.with_cache('medium_cache', ms=3000)
        expected_result = 'howdy'
        call_count = 0

        async def non_caching(o: str, cache_context: ResponseCacheContext[int]) -> str:
            nonlocal call_count
            call_count += 1
            await self.clock.sleep(1)
            cache_context.should_cache = should_cache
            return o
        wrap_d = defer.ensureDeferred(cache.wrap(0, non_caching, expected_result, cache_context=True))
        self.assertNoResult(wrap_d)
        wrap2_d = defer.ensureDeferred(cache.wrap(0, non_caching, expected_result, cache_context=True))
        self.assertNoResult(wrap2_d)
        self.assertEqual(call_count, 1)
        self.reactor.advance(1)
        self.assertEqual(expected_result, self.successResultOf(wrap_d))
        self.assertEqual(expected_result, self.successResultOf(wrap2_d))
        if should_cache:
            unexpected = Mock(spec=())
            wrap3_d = defer.ensureDeferred(cache.wrap(0, unexpected))
            unexpected.assert_not_called()
            self.assertEqual(expected_result, self.successResultOf(wrap3_d), 'cache should still have the result')
        else:
            self.assertCountEqual([], cache.keys(), 'cache should not have the result now')