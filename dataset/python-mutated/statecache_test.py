"""Tests for state caching."""
import logging
import re
import sys
import threading
import time
import unittest
import weakref
import objsize
from hamcrest import assert_that
from hamcrest import contains_string
from apache_beam.runners.worker.statecache import CacheAware
from apache_beam.runners.worker.statecache import StateCache
from apache_beam.runners.worker.statecache import WeightedValue
from apache_beam.runners.worker.statecache import _LoadingValue
from apache_beam.runners.worker.statecache import get_deep_size

class StateCacheTest(unittest.TestCase):

    def test_weakref(self):
        if False:
            return 10
        test_value = WeightedValue('test', 10 << 20)

        class WeightedValueRef:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.ref = weakref.ref(test_value)
        cache = StateCache(5 << 20)
        wait_event = threading.Event()
        o = WeightedValueRef()
        cache.put('deep ref', o)
        self.assertIsNotNone(cache.peek('deep ref'))
        self.assertEqual(cache.describe_stats(), 'used/max 0/5 MB, hit 100.00%, lookups 1, avg load time 0 ns, loads 0, evictions 0')
        cache.invalidate_all()
        o_ref = weakref.ref(o, lambda value: wait_event.set())
        cache.put('not deleted ref', o_ref)
        del o
        wait_event.wait()
        cache.put('deleted', o_ref)

    def test_weakref_proxy(self):
        if False:
            i = 10
            return i + 15
        test_value = WeightedValue('test', 10 << 20)

        class WeightedValueRef:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.ref = weakref.ref(test_value)
        cache = StateCache(5 << 20)
        wait_event = threading.Event()
        o = WeightedValueRef()
        cache.put('deep ref', o)
        self.assertIsNotNone(cache.peek('deep ref'))
        self.assertEqual(cache.describe_stats(), 'used/max 0/5 MB, hit 100.00%, lookups 1, avg load time 0 ns, loads 0, evictions 0')
        cache.invalidate_all()
        o_ref = weakref.proxy(o, lambda value: wait_event.set())
        cache.put('not deleted', o_ref)
        del o
        wait_event.wait()
        cache.put('deleted', o_ref)

    def test_size_of_fails(self):
        if False:
            print('Hello World!')

        class BadSizeOf(object):

            def __sizeof__(self):
                if False:
                    print('Hello World!')
                raise RuntimeError('TestRuntimeError')
        cache = StateCache(5 << 20)
        with self.assertLogs('apache_beam.runners.worker.statecache', level='WARNING') as context:
            cache.put('key', BadSizeOf())
            self.assertEqual(1, len(context.output))
            self.assertTrue('Failed to size' in context.output[0])
            cache.put('key', BadSizeOf())
            self.assertEqual(1, len(context.output))

    def test_empty_cache_peek(self):
        if False:
            print('Hello World!')
        cache = StateCache(5 << 20)
        self.assertEqual(cache.peek('key'), None)
        self.assertEqual(cache.describe_stats(), 'used/max 0/5 MB, hit 0.00%, lookups 1, avg load time 0 ns, loads 0, evictions 0')

    def test_put_peek(self):
        if False:
            return 10
        cache = StateCache(5 << 20)
        cache.put('key', WeightedValue('value', 1 << 20))
        self.assertEqual(cache.size(), 1)
        self.assertEqual(cache.peek('key'), 'value')
        self.assertEqual(cache.peek('key2'), None)
        self.assertEqual(cache.describe_stats(), 'used/max 1/5 MB, hit 50.00%, lookups 2, avg load time 0 ns, loads 0, evictions 0')

    def test_default_sized_put(self):
        if False:
            while True:
                i = 10
        cache = StateCache(5 << 20)
        cache.put('key', bytearray(1 << 20))
        cache.put('key2', bytearray(1 << 20))
        cache.put('key3', bytearray(1 << 20))
        self.assertEqual(cache.peek('key3'), bytearray(1 << 20))
        cache.put('key4', bytearray(1 << 20))
        cache.put('key5', bytearray(1 << 20))
        self.assertEqual(cache.describe_stats(), 'used/max 4/5 MB, hit 100.00%, lookups 1, avg load time 0 ns, loads 0, evictions 1')

    def test_max_size(self):
        if False:
            i = 10
            return i + 15
        cache = StateCache(2 << 20)
        cache.put('key', WeightedValue('value', 1 << 20))
        cache.put('key2', WeightedValue('value2', 1 << 20))
        self.assertEqual(cache.size(), 2)
        cache.put('key3', WeightedValue('value3', 1 << 20))
        self.assertEqual(cache.size(), 2)
        self.assertEqual(cache.describe_stats(), 'used/max 2/2 MB, hit 100.00%, lookups 0, avg load time 0 ns, loads 0, evictions 1')

    def test_invalidate_all(self):
        if False:
            for i in range(10):
                print('nop')
        cache = StateCache(5 << 20)
        cache.put('key', WeightedValue('value', 1 << 20))
        cache.put('key2', WeightedValue('value2', 1 << 20))
        self.assertEqual(cache.size(), 2)
        cache.invalidate_all()
        self.assertEqual(cache.size(), 0)
        self.assertEqual(cache.peek('key'), None)
        self.assertEqual(cache.peek('key2'), None)
        self.assertEqual(cache.describe_stats(), 'used/max 0/5 MB, hit 0.00%, lookups 2, avg load time 0 ns, loads 0, evictions 0')

    def test_lru(self):
        if False:
            print('Hello World!')
        cache = StateCache(5 << 20)
        cache.put('key', WeightedValue('value', 1 << 20))
        cache.put('key2', WeightedValue('value2', 1 << 20))
        cache.put('key3', WeightedValue('value0', 1 << 20))
        cache.put('key3', WeightedValue('value3', 1 << 20))
        cache.put('key4', WeightedValue('value4', 1 << 20))
        cache.put('key5', WeightedValue('value0', 1 << 20))
        cache.put('key5', WeightedValue(['value5'], 1 << 20))
        self.assertEqual(cache.size(), 5)
        self.assertEqual(cache.peek('key'), 'value')
        self.assertEqual(cache.peek('key2'), 'value2')
        self.assertEqual(cache.peek('key3'), 'value3')
        self.assertEqual(cache.peek('key4'), 'value4')
        self.assertEqual(cache.peek('key5'), ['value5'])
        cache.put('key6', WeightedValue('value6', 1 << 20))
        self.assertEqual(cache.size(), 5)
        self.assertEqual(cache.peek('key'), None)
        cache.peek('key2')
        cache.put('key7', WeightedValue('value7', 1 << 20))
        self.assertEqual(cache.size(), 5)
        self.assertEqual(cache.peek('key3'), None)
        cache.put('key8', WeightedValue('put', 1 << 20))
        self.assertEqual(cache.size(), 5)
        cache.put('key9', WeightedValue('value8', 1 << 20))
        self.assertEqual(cache.size(), 5)
        self.assertEqual(cache.peek('key4'), None)
        cache.put('key5', WeightedValue('val', 1 << 20))
        self.assertEqual(cache.peek('key6'), None)
        self.assertEqual(cache.describe_stats(), 'used/max 5/5 MB, hit 60.00%, lookups 10, avg load time 0 ns, loads 0, evictions 5')

    def test_get(self):
        if False:
            while True:
                i = 10

        def check_key(key):
            if False:
                i = 10
                return i + 15
            self.assertEqual(key, 'key')
            time.sleep(0.5)
            return 'value'

        def raise_exception(key):
            if False:
                i = 10
                return i + 15
            time.sleep(0.5)
            raise Exception('TestException')
        cache = StateCache(5 << 20)
        self.assertEqual('value', cache.get('key', check_key))
        with cache._lock:
            self.assertFalse(isinstance(cache._cache['key'], _LoadingValue))
        self.assertEqual('value', cache.peek('key'))
        cache.invalidate_all()
        with self.assertRaisesRegex(Exception, 'TestException'):
            cache.get('key', raise_exception)
        self.assertEqual('value', cache.get('key', check_key))
        with cache._lock:
            self.assertFalse(isinstance(cache._cache['key'], _LoadingValue))
        self.assertEqual('value', cache.peek('key'))
        assert_that(cache.describe_stats(), contains_string(', loads 3,'))
        load_time_ns = re.search(', avg load time (.+) ns,', cache.describe_stats()).group(1)
        self.assertGreater(int(load_time_ns), 0.5 * 1000000000)
        self.assertLess(int(load_time_ns), 1000000000)

    def test_concurrent_get_waits(self):
        if False:
            for i in range(10):
                print('nop')
        event = threading.Semaphore(0)
        threads_running = threading.Barrier(3)

        def wait_for_event(key):
            if False:
                i = 10
                return i + 15
            with cache._lock:
                self.assertTrue(isinstance(cache._cache['key'], _LoadingValue))
            event.release()
            return 'value'
        cache = StateCache(5 << 20)

        def load_key(output):
            if False:
                while True:
                    i = 10
            threads_running.wait()
            output['value'] = cache.get('key', wait_for_event)
            output['time'] = time.time_ns()
        t1_output = {}
        t1 = threading.Thread(target=load_key, args=(t1_output,))
        t1.start()
        t2_output = {}
        t2 = threading.Thread(target=load_key, args=(t2_output,))
        t2.start()
        threads_running.wait()
        current_time_ns = time.time_ns()
        event.acquire()
        t1.join()
        t2.join()
        self.assertFalse(event.acquire(blocking=False))
        self.assertLessEqual(current_time_ns, t1_output['time'])
        self.assertLessEqual(current_time_ns, t2_output['time'])
        self.assertEqual('value', t1_output['value'])
        self.assertEqual('value', t2_output['value'])
        self.assertEqual('value', cache.peek('key'))

    def test_concurrent_get_superseded_by_put(self):
        if False:
            return 10
        load_happening = threading.Event()
        finish_loading = threading.Event()

        def wait_for_event(key):
            if False:
                while True:
                    i = 10
            load_happening.set()
            finish_loading.wait()
            return 'value'
        cache = StateCache(5 << 20)

        def load_key(output):
            if False:
                for i in range(10):
                    print('nop')
            output['value'] = cache.get('key', wait_for_event)
        t1_output = {}
        t1 = threading.Thread(target=load_key, args=(t1_output,))
        t1.start()
        load_happening.wait()
        cache.put('key', 'value2')
        finish_loading.set()
        t1.join()
        self.assertEqual('value', t1_output['value'])
        self.assertEqual('value2', cache.peek('key'))

    def test_is_cached_enabled(self):
        if False:
            i = 10
            return i + 15
        cache = StateCache(1 << 20)
        self.assertEqual(cache.is_cache_enabled(), True)
        self.assertEqual(cache.describe_stats(), 'used/max 0/1 MB, hit 100.00%, lookups 0, avg load time 0 ns, loads 0, evictions 0')
        cache = StateCache(0)
        self.assertEqual(cache.is_cache_enabled(), False)
        self.assertEqual(cache.describe_stats(), 'used/max 0/0 MB, hit 100.00%, lookups 0, avg load time 0 ns, loads 0, evictions 0')

    def test_get_referents_for_cache(self):
        if False:
            print('Hello World!')

        class GetReferentsForCache(CacheAware):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.measure_me = bytearray(1 << 20)
                self.ignore_me = bytearray(2 << 20)

            def get_referents_for_cache(self):
                if False:
                    print('Hello World!')
                return [self.measure_me]
        cache = StateCache(5 << 20)
        cache.put('key', GetReferentsForCache())
        self.assertEqual(cache.describe_stats(), 'used/max 1/5 MB, hit 100.00%, lookups 0, avg load time 0 ns, loads 0, evictions 0')

    def test_get_deep_size_builtin_objects(self):
        if False:
            for i in range(10):
                print('nop')
        '\n    `statecache.get_deep_copy` should work same with objsize unless the `objs`\n    has `CacheAware` or a filtered object. They should return the same size for\n    built-in objects.\n    '
        primitive_test_objects = [1, 2.0, 1 + 1j, True, 'hello,world', b'\x00\x01\x02']
        collection_test_objects = [[3, 4, 5], (6, 7), {'a', 'b', 'c'}, {'k': 8, 'l': 9}]
        for obj in primitive_test_objects:
            self.assertEqual(get_deep_size(obj), objsize.get_deep_size(obj), f'different size for obj: `{obj}`, type: {type(obj)}')
            self.assertEqual(get_deep_size(obj), sys.getsizeof(obj), f'different size for obj: `{obj}`, type: {type(obj)}')
        for obj in collection_test_objects:
            self.assertEqual(get_deep_size(obj), objsize.get_deep_size(obj), f'different size for obj: `{obj}`, type: {type(obj)}')

    def test_current_weight_between_get_and_put(self):
        if False:
            for i in range(10):
                print('nop')
        value = 1234567
        get_cache = StateCache(100)
        get_cache.get('key', lambda k: value)
        put_cache = StateCache(100)
        put_cache.put('key', value)
        self.assertEqual(get_cache._current_weight, put_cache._current_weight)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()