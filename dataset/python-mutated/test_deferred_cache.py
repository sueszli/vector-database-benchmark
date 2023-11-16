from functools import partial
from typing import List, Tuple
from twisted.internet import defer
from synapse.util.caches.deferred_cache import DeferredCache
from tests.unittest import TestCase

class DeferredCacheTestCase(TestCase):

    def test_empty(self) -> None:
        if False:
            return 10
        cache: DeferredCache[str, int] = DeferredCache('test')
        with self.assertRaises(KeyError):
            cache.get('foo')

    def test_hit(self) -> None:
        if False:
            i = 10
            return i + 15
        cache: DeferredCache[str, int] = DeferredCache('test')
        cache.prefill('foo', 123)
        self.assertEqual(self.successResultOf(cache.get('foo')), 123)

    def test_hit_deferred(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cache: DeferredCache[str, int] = DeferredCache('test')
        origin_d: 'defer.Deferred[int]' = defer.Deferred()
        set_d = cache.set('k1', origin_d)
        get_d = cache.get('k1')
        self.assertFalse(get_d.called)

        def check1(r: str) -> str:
            if False:
                i = 10
                return i + 15
            self.assertTrue(set_d.called)
            return r
        get_d.addCallback(check1)
        origin_d.callback(99)
        self.assertEqual(self.successResultOf(origin_d), 99)
        self.assertEqual(self.successResultOf(set_d), 99)
        self.assertEqual(self.successResultOf(get_d), 99)

    def test_callbacks(self) -> None:
        if False:
            while True:
                i = 10
        'Invalidation callbacks are called at the right time'
        cache: DeferredCache[str, int] = DeferredCache('test')
        callbacks = set()
        cache.prefill('k1', 10, callback=lambda : callbacks.add('prefill'))
        origin_d: 'defer.Deferred[int]' = defer.Deferred()
        set_d = cache.set('k1', origin_d, callback=lambda : callbacks.add('set'))
        get_d = cache.get('k1', callback=lambda : callbacks.add('get'))
        self.assertEqual(callbacks, set())
        origin_d.callback(20)
        self.assertEqual(self.successResultOf(set_d), 20)
        self.assertEqual(self.successResultOf(get_d), 20)
        self.assertEqual(callbacks, {'prefill'})
        callbacks.clear()
        cache.prefill('k1', 30)
        self.assertEqual(callbacks, {'set', 'get'})

    def test_set_fail(self) -> None:
        if False:
            return 10
        cache: DeferredCache[str, int] = DeferredCache('test')
        callbacks = set()
        cache.prefill('k1', 10, callback=lambda : callbacks.add('prefill'))
        origin_d: defer.Deferred = defer.Deferred()
        set_d = cache.set('k1', origin_d, callback=lambda : callbacks.add('set'))
        get_d = cache.get('k1', callback=lambda : callbacks.add('get'))
        self.assertEqual(callbacks, set())
        e = Exception('oops')
        origin_d.errback(e)
        self.assertIs(self.failureResultOf(set_d, Exception).value, e)
        self.assertIs(self.failureResultOf(get_d, Exception).value, e)
        self.assertEqual(callbacks, {'get', 'set'})
        callbacks.clear()
        get_d2 = cache.get('k1', callback=lambda : callbacks.add('get2'))
        self.assertEqual(self.successResultOf(get_d2), 10)
        cache.prefill('k1', 30)
        self.assertEqual(callbacks, {'prefill', 'get2'})

    def test_get_immediate(self) -> None:
        if False:
            i = 10
            return i + 15
        cache: DeferredCache[str, int] = DeferredCache('test')
        d1: 'defer.Deferred[int]' = defer.Deferred()
        cache.set('key1', d1)
        v = cache.get_immediate('key1', 1)
        self.assertEqual(v, 1)
        d1.callback(2)
        v = cache.get_immediate('key1', 1)
        self.assertEqual(v, 2)

    def test_invalidate(self) -> None:
        if False:
            while True:
                i = 10
        cache: DeferredCache[Tuple[str], int] = DeferredCache('test')
        cache.prefill(('foo',), 123)
        cache.invalidate(('foo',))
        with self.assertRaises(KeyError):
            cache.get(('foo',))

    def test_invalidate_all(self) -> None:
        if False:
            return 10
        cache: DeferredCache[str, str] = DeferredCache('testcache')
        callback_record = [False, False]

        def record_callback(idx: int) -> None:
            if False:
                i = 10
                return i + 15
            callback_record[idx] = True
        d1: 'defer.Deferred[str]' = defer.Deferred()
        cache.set('key1', d1, partial(record_callback, 0))
        d2: 'defer.Deferred[str]' = defer.Deferred()
        cache.set('key2', d2, partial(record_callback, 1))
        self.assertFalse(cache.get('key1').called)
        self.assertFalse(cache.get('key2').called)
        d2.callback('result2')
        self.assertEqual(self.successResultOf(cache.get('key2')), 'result2')
        cache.invalidate_all()
        with self.assertRaises(KeyError):
            cache.get('key1')
        with self.assertRaises(KeyError):
            cache.get('key2')
        self.assertTrue(callback_record[0], 'Invalidation callback for key1 not called')
        self.assertTrue(callback_record[1], 'Invalidation callback for key2 not called')
        d1.callback('result1')
        with self.assertRaises(KeyError):
            cache.get('key1', None)

    def test_eviction(self) -> None:
        if False:
            i = 10
            return i + 15
        cache: DeferredCache[int, str] = DeferredCache('test', max_entries=2, apply_cache_factor_from_config=False)
        cache.prefill(1, 'one')
        cache.prefill(2, 'two')
        cache.prefill(3, 'three')
        with self.assertRaises(KeyError):
            cache.get(1)
        cache.get(2)
        cache.get(3)

    def test_eviction_lru(self) -> None:
        if False:
            print('Hello World!')
        cache: DeferredCache[int, str] = DeferredCache('test', max_entries=2, apply_cache_factor_from_config=False)
        cache.prefill(1, 'one')
        cache.prefill(2, 'two')
        cache.get(1)
        cache.prefill(3, 'three')
        with self.assertRaises(KeyError):
            cache.get(2)
        cache.get(1)
        cache.get(3)

    def test_eviction_iterable(self) -> None:
        if False:
            print('Hello World!')
        cache: DeferredCache[int, List[str]] = DeferredCache('test', max_entries=3, apply_cache_factor_from_config=False, iterable=True)
        cache.prefill(1, ['one', 'two'])
        cache.prefill(2, ['three'])
        cache.get(1)
        cache.prefill(3, ['four'])
        with self.assertRaises(KeyError):
            cache.get(2)
        cache.get(1)
        cache.get(3)
        cache.get(1)
        cache.prefill(4, ['five', 'six'])
        with self.assertRaises(KeyError):
            cache.get(1)
        with self.assertRaises(KeyError):
            cache.get(3)
        cache.prefill(5, ['seven'])
        cache.get(4)
        cache.prefill(6, [])
        with self.assertRaises(KeyError):
            cache.get(5)
        cache.get(4)
        cache.get(6)