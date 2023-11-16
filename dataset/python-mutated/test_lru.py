import gc
import random
import string
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import failure
from twisted.trial import unittest
from buildbot.util import lru

def short(k):
    if False:
        for i in range(10):
            print('nop')
    return set([k.upper() * 3])

def long(k):
    if False:
        print('Hello World!')
    return set([k.upper() * 6])

class LRUCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        lru.inv_failed = False
        self.lru = lru.LRUCache(short, 3)

    def tearDown(self):
        if False:
            return 10
        self.assertFalse(lru.inv_failed, 'invariant failed; see logs')

    def check_result(self, r, exp, exp_hits=None, exp_misses=None, exp_refhits=None):
        if False:
            while True:
                i = 10
        self.assertEqual(r, exp)
        if exp_hits is not None:
            self.assertEqual(self.lru.hits, exp_hits)
        if exp_misses is not None:
            self.assertEqual(self.lru.misses, exp_misses)
        if exp_refhits is not None:
            self.assertEqual(self.lru.refhits, exp_refhits)

    def test_single_key(self):
        if False:
            return 10
        val = self.lru.get('a')
        self.check_result(val, short('a'), 0, 1)
        self.lru.miss_fn = long
        val = self.lru.get('a')
        self.check_result(val, short('a'), 1, 1)

    def test_simple_lru_expulsion(self):
        if False:
            return 10
        val = self.lru.get('a')
        self.check_result(val, short('a'), 0, 1)
        val = self.lru.get('b')
        self.check_result(val, short('b'), 0, 2)
        val = self.lru.get('c')
        self.check_result(val, short('c'), 0, 3)
        val = self.lru.get('d')
        self.check_result(val, short('d'), 0, 4)
        del val
        gc.collect()
        self.lru.miss_fn = long
        val = self.lru.get('a')
        self.check_result(val, long('a'), 0, 5)
        val = self.lru.get('c')
        self.check_result(val, short('c'), 1, 5)

    @defer.inlineCallbacks
    def test_simple_lru_expulsion_maxsize_1(self):
        if False:
            return 10
        self.lru = lru.LRUCache(short, 1)
        val = (yield self.lru.get('a'))
        self.check_result(val, short('a'), 0, 1)
        val = (yield self.lru.get('a'))
        self.check_result(val, short('a'), 1, 1)
        val = (yield self.lru.get('b'))
        self.check_result(val, short('b'), 1, 2)
        del val
        gc.collect()
        self.lru.miss_fn = long
        val = (yield self.lru.get('a'))
        self.check_result(val, long('a'), 1, 3)
        del val
        gc.collect()
        val = (yield self.lru.get('b'))
        self.check_result(val, long('b'), 1, 4)

    def test_simple_lru_expulsion_maxsize_1_null_result(self):
        if False:
            for i in range(10):
                print('nop')

        def miss_fn(k):
            if False:
                return 10
            if k == 'b':
                return None
            return short(k)
        self.lru = lru.LRUCache(miss_fn, 1)
        val = self.lru.get('a')
        self.check_result(val, short('a'), 0, 1)
        val = self.lru.get('b')
        self.check_result(val, None, 0, 2)
        del val
        self.lru.miss_fn = long
        val = self.lru.get('a')
        self.check_result(val, short('a'), 1, 2)

    def test_queue_collapsing(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.lru.max_queue, 30)
        for c in 'a' + 'x' * 27 + 'ab':
            res = self.lru.get(c)
        self.check_result(res, short('b'), 27, 3)
        self.assertEqual(len(self.lru.queue), 30)
        res = self.lru.get('b')
        self.check_result(res, short('b'), 28, 3)
        self.assertEqual(len(self.lru.queue), 3)
        self.lru.miss_fn = long
        res = self.lru.get('a')
        self.check_result(res, short('a'), 29, 3)

    def test_all_misses(self):
        if False:
            print('Hello World!')
        for (i, c) in enumerate(string.ascii_lowercase + string.ascii_uppercase):
            res = self.lru.get(c)
            self.check_result(res, short(c), 0, i + 1)

    def test_get_exception(self):
        if False:
            print('Hello World!')

        def fail_miss_fn(k):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('oh noes')
        self.lru.miss_fn = fail_miss_fn
        got_exc = False
        try:
            self.lru.get('abc')
        except RuntimeError:
            got_exc = True
        self.assertEqual(got_exc, True)

    def test_all_hits(self):
        if False:
            i = 10
            return i + 15
        res = self.lru.get('a')
        self.check_result(res, short('a'), 0, 1)
        self.lru.miss_fn = long
        for i in range(100):
            res = self.lru.get('a')
            self.check_result(res, short('a'), i + 1, 1)

    def test_weakrefs(self):
        if False:
            print('Hello World!')
        res_a = self.lru.get('a')
        self.check_result(res_a, short('a'))
        res_b = self.lru.get('b')
        self.check_result(res_b, short('b'))
        del res_b
        self.lru.miss_fn = long
        for c in string.ascii_lowercase[2:] * 5:
            self.lru.get(c)
        res = self.lru.get('a')
        self.check_result(res, res_a, exp_refhits=1)
        res = self.lru.get('b')
        self.check_result(res, long('b'), exp_refhits=1)

    def test_fuzz(self):
        if False:
            for i in range(10):
                print('nop')
        chars = list(string.ascii_lowercase * 40)
        random.shuffle(chars)
        for c in chars:
            res = self.lru.get(c)
            self.check_result(res, short(c))

    def test_set_max_size(self):
        if False:
            for i in range(10):
                print('nop')
        for c in 'abc':
            res = self.lru.get(c)
            self.check_result(res, short(c))
        del res
        self.lru.set_max_size(1)
        gc.collect()
        self.lru.miss_fn = long
        res = self.lru.get('b')
        self.check_result(res, long('b'))

    def test_miss_fn_kwargs(self):
        if False:
            print('Hello World!')

        def keep_kwargs_miss_fn(k, **kwargs):
            if False:
                i = 10
                return i + 15
            return set(kwargs.keys())
        self.lru.miss_fn = keep_kwargs_miss_fn
        val = self.lru.get('a', a=1, b=2)
        self.check_result(val, set(['a', 'b']), 0, 1)

    def test_miss_fn_returns_none(self):
        if False:
            while True:
                i = 10
        calls = []

        def none_miss_fn(k):
            if False:
                return 10
            calls.append(k)
            return None
        self.lru.miss_fn = none_miss_fn
        for _ in range(2):
            self.assertEqual(self.lru.get('a'), None)
        self.assertEqual(calls, ['a', 'a'])

    def test_put(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.lru.get('p'), short('p'))
        self.lru.put('p', set(['P2P2']))
        self.assertEqual(self.lru.get('p'), set(['P2P2']))

    def test_put_nonexistent_key(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.lru.get('p'), short('p'))
        self.lru.put('q', set(['new-q']))
        self.assertEqual(self.lru.get('p'), set(['PPP']))
        self.assertEqual(self.lru.get('q'), set(['new-q']))

class AsyncLRUCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        lru.inv_failed = False
        self.lru = lru.AsyncLRUCache(self.short_miss_fn, 3)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.assertFalse(lru.inv_failed, 'invariant failed; see logs')

    def short_miss_fn(self, key):
        if False:
            return 10
        return defer.succeed(short(key))

    def long_miss_fn(self, key):
        if False:
            print('Hello World!')
        return defer.succeed(long(key))

    def failure_miss_fn(self, key):
        if False:
            return 10
        return defer.succeed(None)

    def check_result(self, r, exp, exp_hits=None, exp_misses=None, exp_refhits=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(r, exp)
        if exp_hits is not None:
            self.assertEqual(self.lru.hits, exp_hits)
        if exp_misses is not None:
            self.assertEqual(self.lru.misses, exp_misses)
        if exp_refhits is not None:
            self.assertEqual(self.lru.refhits, exp_refhits)

    @defer.inlineCallbacks
    def test_single_key(self):
        if False:
            return 10
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 0, 1)
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 1, 1)

    @defer.inlineCallbacks
    def test_simple_lru_expulsion(self):
        if False:
            return 10
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 0, 1)
        res = (yield self.lru.get('b'))
        self.check_result(res, short('b'), 0, 2)
        res = (yield self.lru.get('c'))
        self.check_result(res, short('c'), 0, 3)
        res = (yield self.lru.get('d'))
        self.check_result(res, short('d'), 0, 4)
        gc.collect()
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('a'))
        self.check_result(res, long('a'), 0, 5)
        res = (yield self.lru.get('c'))
        self.check_result(res, short('c'), 1, 5)

    @defer.inlineCallbacks
    def test_simple_lru_expulsion_maxsize_1(self):
        if False:
            while True:
                i = 10
        self.lru = lru.AsyncLRUCache(self.short_miss_fn, 1)
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 0, 1)
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 1, 1)
        res = (yield self.lru.get('b'))
        self.check_result(res, short('b'), 1, 2)
        gc.collect()
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('a'))
        self.check_result(res, long('a'), 1, 3)
        gc.collect()
        res = (yield self.lru.get('b'))
        self.check_result(res, long('b'), 1, 4)

    @defer.inlineCallbacks
    def test_simple_lru_expulsion_maxsize_1_null_result(self):
        if False:
            for i in range(10):
                print('nop')

        def miss_fn(k):
            if False:
                while True:
                    i = 10
            if k == 'b':
                return defer.succeed(None)
            return defer.succeed(short(k))
        self.lru = lru.AsyncLRUCache(miss_fn, 1)
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 0, 1)
        res = (yield self.lru.get('b'))
        self.check_result(res, None, 0, 2)
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 1, 2)

    @defer.inlineCallbacks
    def test_queue_collapsing(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.lru.max_queue, 30)
        for c in 'a' + 'x' * 27 + 'ab':
            res = (yield self.lru.get(c))
        self.check_result(res, short('b'), 27, 3)
        self.assertEqual(len(self.lru.queue), 30)
        res = (yield self.lru.get('b'))
        self.check_result(res, short('b'), 28, 3)
        self.assertEqual(len(self.lru.queue), 3)
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 29, 3)

    @defer.inlineCallbacks
    def test_all_misses(self):
        if False:
            i = 10
            return i + 15
        for (i, c) in enumerate(string.ascii_lowercase + string.ascii_uppercase):
            res = (yield self.lru.get(c))
            self.check_result(res, short(c), 0, i + 1)

    @defer.inlineCallbacks
    def test_get_exception(self):
        if False:
            return 10

        def fail_miss_fn(k):
            if False:
                return 10
            return defer.fail(RuntimeError('oh noes'))
        self.lru.miss_fn = fail_miss_fn
        got_exc = False
        try:
            yield self.lru.get('abc')
        except RuntimeError:
            got_exc = True
        self.assertEqual(got_exc, True)

    @defer.inlineCallbacks
    def test_all_hits(self):
        if False:
            while True:
                i = 10
        res = (yield self.lru.get('a'))
        self.check_result(res, short('a'), 0, 1)
        self.lru.miss_fn = self.long_miss_fn
        for i in range(100):
            res = (yield self.lru.get('a'))
            self.check_result(res, short('a'), i + 1, 1)

    @defer.inlineCallbacks
    def test_weakrefs(self):
        if False:
            return 10
        res_a = (yield self.lru.get('a'))
        self.check_result(res_a, short('a'))
        res_b = (yield self.lru.get('b'))
        self.check_result(res_b, short('b'))
        del res_b
        self.lru.miss_fn = self.long_miss_fn
        for c in string.ascii_lowercase[2:] * 5:
            yield self.lru.get(c)
        res = (yield self.lru.get('a'))
        self.check_result(res, res_a, exp_refhits=1)
        res = (yield self.lru.get('b'))
        self.check_result(res, long('b'), exp_refhits=1)

    @defer.inlineCallbacks
    def test_fuzz(self):
        if False:
            for i in range(10):
                print('nop')
        chars = list(string.ascii_lowercase * 40)
        random.shuffle(chars)
        for c in chars:
            res = (yield self.lru.get(c))
            self.check_result(res, short(c))

    @defer.inlineCallbacks
    def test_massively_parallel(self):
        if False:
            print('Hello World!')
        chars = list(string.ascii_lowercase * 5)
        misses = [0]

        def slow_short_miss_fn(key):
            if False:
                while True:
                    i = 10
            d = defer.Deferred()
            misses[0] += 1
            reactor.callLater(0, lambda : d.callback(short(key)))
            return d
        self.lru.miss_fn = slow_short_miss_fn

        def check(c, d):
            if False:
                while True:
                    i = 10
            d.addCallback(self.check_result, short(c))
            return d
        yield defer.gatherResults([check(c, self.lru.get(c)) for c in chars])
        self.assertEqual(misses[0], 26)
        self.assertEqual(self.lru.misses, 26)
        self.assertEqual(self.lru.hits, 4 * 26)

    @defer.inlineCallbacks
    def test_slow_fetch(self):
        if False:
            while True:
                i = 10

        def slower_miss_fn(k):
            if False:
                print('Hello World!')
            d = defer.Deferred()
            reactor.callLater(0.05, lambda : d.callback(short(k)))
            return d
        self.lru.miss_fn = slower_miss_fn

        def do_get(test_d, k):
            if False:
                while True:
                    i = 10
            d = self.lru.get(k)
            d.addCallback(self.check_result, short(k))
            d.addCallbacks(test_d.callback, test_d.errback)
        ds = []
        for i in range(8):
            d = defer.Deferred()
            reactor.callLater(0.02 * i, do_get, d, 'x')
            ds.append(d)
        yield defer.gatherResults(ds)
        self.assertEqual((self.lru.hits, self.lru.misses), (7, 1))

    def test_slow_failure(self):
        if False:
            i = 10
            return i + 15

        def slow_fail_miss_fn(k):
            if False:
                return 10
            d = defer.Deferred()
            reactor.callLater(0.05, lambda : d.errback(failure.Failure(RuntimeError('oh noes'))))
            return d
        self.lru.miss_fn = slow_fail_miss_fn

        @defer.inlineCallbacks
        def do_get(test_d, k):
            if False:
                return 10
            d = self.lru.get(k)
            yield self.assertFailure(d, RuntimeError)
            d.addCallbacks(test_d.callback, test_d.errback)
        ds = []
        for i in range(8):
            d = defer.Deferred()
            reactor.callLater(0.02 * i, do_get, d, 'x')
            ds.append(d)
        d = defer.gatherResults(ds)
        return d

    @defer.inlineCallbacks
    def test_set_max_size(self):
        if False:
            while True:
                i = 10
        for c in 'abc':
            res = (yield self.lru.get(c))
            self.check_result(res, short(c))
        self.lru.set_max_size(1)
        gc.collect()
        self.lru.miss_fn = self.long_miss_fn
        res = (yield self.lru.get('b'))
        self.check_result(res, long('b'))

    @defer.inlineCallbacks
    def test_miss_fn_kwargs(self):
        if False:
            i = 10
            return i + 15

        def keep_kwargs_miss_fn(k, **kwargs):
            if False:
                i = 10
                return i + 15
            return defer.succeed(set(kwargs.keys()))
        self.lru.miss_fn = keep_kwargs_miss_fn
        res = (yield self.lru.get('a', a=1, b=2))
        self.check_result(res, set(['a', 'b']), 0, 1)

    @defer.inlineCallbacks
    def test_miss_fn_returns_none(self):
        if False:
            while True:
                i = 10
        calls = []

        def none_miss_fn(k):
            if False:
                for i in range(10):
                    print('nop')
            calls.append(k)
            return defer.succeed(None)
        self.lru.miss_fn = none_miss_fn
        for _ in range(2):
            self.assertEqual((yield self.lru.get('a')), None)
        self.assertEqual(calls, ['a', 'a'])

    @defer.inlineCallbacks
    def test_put(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual((yield self.lru.get('p')), short('p'))
        self.lru.put('p', set(['P2P2']))
        self.assertEqual((yield self.lru.get('p')), set(['P2P2']))