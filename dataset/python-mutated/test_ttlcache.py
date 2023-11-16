from unittest.mock import Mock
from synapse.util.caches.ttlcache import TTLCache
from tests import unittest

class CacheTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.mock_timer = Mock(side_effect=lambda : 100.0)
        self.cache: TTLCache[str, str] = TTLCache('test_cache', self.mock_timer)

    def test_get(self) -> None:
        if False:
            return 10
        'simple set/get tests'
        self.cache.set('one', '1', 10)
        self.cache.set('two', '2', 20)
        self.cache.set('three', '3', 30)
        self.assertEqual(len(self.cache), 3)
        self.assertTrue('one' in self.cache)
        self.assertEqual(self.cache.get('one'), '1')
        self.assertEqual(self.cache['one'], '1')
        self.assertEqual(self.cache.get_with_expiry('one'), ('1', 110, 10))
        self.assertEqual(self.cache._metrics.hits, 3)
        self.assertEqual(self.cache._metrics.misses, 0)
        self.cache.set('two', '2.5', 20)
        self.assertEqual(self.cache['two'], '2.5')
        self.assertEqual(self.cache._metrics.hits, 4)
        self.assertEqual(self.cache.get('four', '4'), '4')
        self.assertIs(self.cache.get('four', None), None)
        with self.assertRaises(KeyError):
            self.cache['four']
        with self.assertRaises(KeyError):
            self.cache.get('four')
        with self.assertRaises(KeyError):
            self.cache.get_with_expiry('four')
        self.assertEqual(self.cache._metrics.hits, 4)
        self.assertEqual(self.cache._metrics.misses, 5)

    def test_expiry(self) -> None:
        if False:
            print('Hello World!')
        self.cache.set('one', '1', 10)
        self.cache.set('two', '2', 20)
        self.cache.set('three', '3', 30)
        self.assertEqual(len(self.cache), 3)
        self.assertEqual(self.cache['one'], '1')
        self.assertEqual(self.cache['two'], '2')
        self.mock_timer.side_effect = lambda : 110.0
        self.assertEqual(len(self.cache), 2)
        self.assertFalse('one' in self.cache)
        self.assertEqual(self.cache['two'], '2')
        self.assertEqual(self.cache['three'], '3')
        self.assertEqual(self.cache.get_with_expiry('two'), ('2', 120, 20))
        self.assertEqual(self.cache._metrics.hits, 5)
        self.assertEqual(self.cache._metrics.misses, 0)