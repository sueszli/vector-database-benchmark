from synapse.util.caches.dictionary_cache import DictionaryCache
from tests import unittest

class DictCacheTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.cache: DictionaryCache[str, str, str] = DictionaryCache('foobar', max_entries=10)

    def test_simple_cache_hit_full(self) -> None:
        if False:
            while True:
                i = 10
        key = 'test_simple_cache_hit_full'
        v = self.cache.get(key)
        self.assertIs(v.full, False)
        self.assertEqual(v.known_absent, set())
        self.assertEqual({}, v.value)
        seq = self.cache.sequence
        test_value = {'test': 'test_simple_cache_hit_full'}
        self.cache.update(seq, key, test_value)
        c = self.cache.get(key)
        self.assertEqual(test_value, c.value)

    def test_simple_cache_hit_partial(self) -> None:
        if False:
            print('Hello World!')
        key = 'test_simple_cache_hit_partial'
        seq = self.cache.sequence
        test_value = {'test': 'test_simple_cache_hit_partial'}
        self.cache.update(seq, key, test_value)
        c = self.cache.get(key, ['test'])
        self.assertEqual(test_value, c.value)

    def test_simple_cache_miss_partial(self) -> None:
        if False:
            while True:
                i = 10
        key = 'test_simple_cache_miss_partial'
        seq = self.cache.sequence
        test_value = {'test': 'test_simple_cache_miss_partial'}
        self.cache.update(seq, key, test_value)
        c = self.cache.get(key, ['test2'])
        self.assertEqual({}, c.value)

    def test_simple_cache_hit_miss_partial(self) -> None:
        if False:
            while True:
                i = 10
        key = 'test_simple_cache_hit_miss_partial'
        seq = self.cache.sequence
        test_value = {'test': 'test_simple_cache_hit_miss_partial', 'test2': 'test_simple_cache_hit_miss_partial2', 'test3': 'test_simple_cache_hit_miss_partial3'}
        self.cache.update(seq, key, test_value)
        c = self.cache.get(key, ['test2'])
        self.assertEqual({'test2': 'test_simple_cache_hit_miss_partial2'}, c.value)

    def test_multi_insert(self) -> None:
        if False:
            while True:
                i = 10
        key = 'test_simple_cache_hit_miss_partial'
        seq = self.cache.sequence
        test_value_1 = {'test': 'test_simple_cache_hit_miss_partial'}
        self.cache.update(seq, key, test_value_1, fetched_keys={'test'})
        seq = self.cache.sequence
        test_value_2 = {'test2': 'test_simple_cache_hit_miss_partial2'}
        self.cache.update(seq, key, test_value_2, fetched_keys={'test2'})
        c = self.cache.get(key, dict_keys=['test', 'test2'])
        self.assertEqual({'test': 'test_simple_cache_hit_miss_partial', 'test2': 'test_simple_cache_hit_miss_partial2'}, c.value)
        self.assertEqual(c.full, False)

    def test_invalidation(self) -> None:
        if False:
            while True:
                i = 10
        'Test that the partial dict and full dicts get invalidated\n        separately.\n        '
        key = 'some_key'
        seq = self.cache.sequence
        self.cache.update(seq, key, {'a': 'b', 'c': 'd'})
        for i in range(20):
            self.cache.get(key, ['a'])
            self.cache.update(seq, f'key{i}', {'1': '2'})
        r = self.cache.get(key)
        self.assertFalse(r.full)
        self.assertTrue('c' not in r.value)
        r = self.cache.get(key, dict_keys=['a'])
        self.assertFalse(r.full)
        self.assertEqual(r.value, {'a': 'b'})