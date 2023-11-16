"""Tests for the fifo_cache module."""
from bzrlib import fifo_cache, tests

class TestFIFOCache(tests.TestCase):
    """Test that FIFO cache properly keeps track of entries."""

    def test_add_is_present(self):
        if False:
            while True:
                i = 10
        c = fifo_cache.FIFOCache()
        c[1] = 2
        self.assertTrue(1 in c)
        self.assertEqual(1, len(c))
        self.assertEqual(2, c[1])
        self.assertEqual(2, c.get(1))
        self.assertEqual(2, c.get(1, None))
        self.assertEqual([1], c.keys())
        self.assertEqual([1], list(c.iterkeys()))
        self.assertEqual([(1, 2)], c.items())
        self.assertEqual([(1, 2)], list(c.iteritems()))
        self.assertEqual([2], c.values())
        self.assertEqual([2], list(c.itervalues()))
        self.assertEqual({1: 2}, c)

    def test_cache_size(self):
        if False:
            print('Hello World!')
        c = fifo_cache.FIFOCache()
        self.assertEqual(100, c.cache_size())
        c.resize(20, 5)
        self.assertEqual(20, c.cache_size())

    def test_missing(self):
        if False:
            print('Hello World!')
        c = fifo_cache.FIFOCache()
        self.assertRaises(KeyError, c.__getitem__, 1)
        self.assertFalse(1 in c)
        self.assertEqual(0, len(c))
        self.assertEqual(None, c.get(1))
        self.assertEqual(None, c.get(1, None))
        self.assertEqual([], c.keys())
        self.assertEqual([], list(c.iterkeys()))
        self.assertEqual([], c.items())
        self.assertEqual([], list(c.iteritems()))
        self.assertEqual([], c.values())
        self.assertEqual([], list(c.itervalues()))
        self.assertEqual({}, c)

    def test_add_maintains_fifo(self):
        if False:
            while True:
                i = 10
        c = fifo_cache.FIFOCache(4, 4)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        self.assertEqual([1, 2, 3, 4], sorted(c.keys()))
        c[5] = 6
        self.assertEqual([2, 3, 4, 5], sorted(c.keys()))
        c[2] = 7
        self.assertEqual([2, 3, 4, 5], sorted(c.keys()))
        c[6] = 7
        self.assertEqual([2, 4, 5, 6], sorted(c.keys()))
        self.assertEqual([4, 5, 2, 6], list(c._queue))

    def test_default_after_cleanup_count(self):
        if False:
            while True:
                i = 10
        c = fifo_cache.FIFOCache(5)
        self.assertEqual(4, c._after_cleanup_count)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        self.assertEqual([1, 2, 3, 4, 5], sorted(c.keys()))
        c[6] = 7
        self.assertEqual([3, 4, 5, 6], sorted(c.keys()))

    def test_clear(self):
        if False:
            for i in range(10):
                print('nop')
        c = fifo_cache.FIFOCache(5)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.cleanup()
        self.assertEqual([2, 3, 4, 5], sorted(c.keys()))
        c.clear()
        self.assertEqual([], c.keys())
        self.assertEqual([], list(c._queue))
        self.assertEqual({}, c)

    def test_copy_not_implemented(self):
        if False:
            i = 10
            return i + 15
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.copy)

    def test_pop_not_implemeted(self):
        if False:
            return 10
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.pop, 'key')

    def test_popitem_not_implemeted(self):
        if False:
            return 10
        c = fifo_cache.FIFOCache()
        self.assertRaises(NotImplementedError, c.popitem)

    def test_resize_smaller(self):
        if False:
            i = 10
            return i + 15
        c = fifo_cache.FIFOCache()
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.resize(5)
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, c)
        self.assertEqual(5, c.cache_size())
        c[6] = 7
        self.assertEqual({3: 4, 4: 5, 5: 6, 6: 7}, c)
        c.resize(3, 2)
        self.assertEqual({5: 6, 6: 7}, c)

    def test_resize_larger(self):
        if False:
            print('Hello World!')
        c = fifo_cache.FIFOCache(5, 4)
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5
        c[5] = 6
        c.resize(10)
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, c)
        self.assertEqual(10, c.cache_size())
        c[6] = 7
        c[7] = 8
        c[8] = 9
        c[9] = 10
        c[10] = 11
        self.assertEqual({1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}, c)
        c[11] = 12
        self.assertEqual({4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12}, c)

    def test_setdefault(self):
        if False:
            for i in range(10):
                print('nop')
        c = fifo_cache.FIFOCache(5, 4)
        c['one'] = 1
        c['two'] = 2
        c['three'] = 3
        myobj = object()
        self.assertIs(myobj, c.setdefault('four', myobj))
        self.assertEqual({'one': 1, 'two': 2, 'three': 3, 'four': myobj}, c)
        self.assertEqual(3, c.setdefault('three', myobj))
        c.setdefault('five', myobj)
        c.setdefault('six', myobj)
        self.assertEqual({'three': 3, 'four': myobj, 'five': myobj, 'six': myobj}, c)

    def test_update(self):
        if False:
            while True:
                i = 10
        c = fifo_cache.FIFOCache(5, 4)
        c.update([(1, 2), (3, 4)])
        self.assertEqual({1: 2, 3: 4}, c)
        c.update(foo=3, bar=4)
        self.assertEqual({1: 2, 3: 4, 'foo': 3, 'bar': 4}, c)
        c.update({'baz': 'biz', 'bing': 'bang'})
        self.assertEqual({'foo': 3, 'bar': 4, 'baz': 'biz', 'bing': 'bang'}, c)
        self.assertRaises(TypeError, c.update, [(1, 2)], [(3, 4)])
        c.update([('a', 'b'), ('d', 'e')], a='c', q='r')
        self.assertEqual({'baz': 'biz', 'bing': 'bang', 'a': 'c', 'd': 'e', 'q': 'r'}, c)

    def test_cleanup_funcs(self):
        if False:
            while True:
                i = 10
        log = []

        def logging_cleanup(key, value):
            if False:
                while True:
                    i = 10
            log.append((key, value))
        c = fifo_cache.FIFOCache(5, 4)
        c.add(1, 2, cleanup=logging_cleanup)
        c.add(2, 3, cleanup=logging_cleanup)
        c.add(3, 4, cleanup=logging_cleanup)
        c.add(4, 5, cleanup=None)
        c[5] = 6
        self.assertEqual([], log)
        c.add(6, 7, cleanup=logging_cleanup)
        self.assertEqual([(1, 2), (2, 3)], log)
        del log[:]
        c.add(3, 8, cleanup=logging_cleanup)
        self.assertEqual([(3, 4)], log)
        del log[:]
        c[3] = 9
        self.assertEqual([(3, 8)], log)
        del log[:]
        c.clear()
        self.assertEqual([(6, 7)], log)
        del log[:]
        c.add(8, 9, cleanup=logging_cleanup)
        del c[8]
        self.assertEqual([(8, 9)], log)

    def test_cleanup_at_deconstruct(self):
        if False:
            print('Hello World!')
        log = []

        def logging_cleanup(key, value):
            if False:
                i = 10
                return i + 15
            log.append((key, value))
        c = fifo_cache.FIFOCache()
        c.add(1, 2, cleanup=logging_cleanup)
        del c
        self.assertEqual([], log)

class TestFIFOSizeCache(tests.TestCase):

    def test_add_is_present(self):
        if False:
            print('Hello World!')
        c = fifo_cache.FIFOSizeCache()
        c[1] = '2'
        self.assertTrue(1 in c)
        self.assertEqual(1, len(c))
        self.assertEqual('2', c[1])
        self.assertEqual('2', c.get(1))
        self.assertEqual('2', c.get(1, None))
        self.assertEqual([1], c.keys())
        self.assertEqual([1], list(c.iterkeys()))
        self.assertEqual([(1, '2')], c.items())
        self.assertEqual([(1, '2')], list(c.iteritems()))
        self.assertEqual(['2'], c.values())
        self.assertEqual(['2'], list(c.itervalues()))
        self.assertEqual({1: '2'}, c)
        self.assertEqual(1024 * 1024, c.cache_size())

    def test_missing(self):
        if False:
            print('Hello World!')
        c = fifo_cache.FIFOSizeCache()
        self.assertRaises(KeyError, c.__getitem__, 1)
        self.assertFalse(1 in c)
        self.assertEqual(0, len(c))
        self.assertEqual(None, c.get(1))
        self.assertEqual(None, c.get(1, None))
        self.assertEqual([], c.keys())
        self.assertEqual([], list(c.iterkeys()))
        self.assertEqual([], c.items())
        self.assertEqual([], list(c.iteritems()))
        self.assertEqual([], c.values())
        self.assertEqual([], list(c.itervalues()))
        self.assertEqual({}, c)

    def test_add_maintains_fifo(self):
        if False:
            for i in range(10):
                print('nop')
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'ab'
        c[2] = 'cde'
        c[3] = 'fghi'
        self.assertEqual({1: 'ab', 2: 'cde', 3: 'fghi'}, c)
        c[4] = 'jkl'
        self.assertEqual({3: 'fghi', 4: 'jkl'}, c)
        c[3] = 'mnop'
        self.assertEqual({3: 'mnop', 4: 'jkl'}, c)
        c[5] = 'qrst'
        self.assertEqual({3: 'mnop', 5: 'qrst'}, c)

    def test_adding_large_key(self):
        if False:
            return 10
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'abcdefgh'
        self.assertEqual({}, c)
        c[1] = 'abcdefg'
        self.assertEqual({1: 'abcdefg'}, c)
        c[1] = 'abcdefgh'
        self.assertEqual({}, c)
        self.assertEqual(0, c._value_size)

    def test_resize_smaller(self):
        if False:
            i = 10
            return i + 15
        c = fifo_cache.FIFOSizeCache(20, 16)
        c[1] = 'a'
        c[2] = 'bc'
        c[3] = 'def'
        c[4] = 'ghij'
        c.resize(10, 8)
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij'}, c)
        self.assertEqual(10, c.cache_size())
        c[5] = 'k'
        self.assertEqual({3: 'def', 4: 'ghij', 5: 'k'}, c)
        c.resize(5, 4)
        self.assertEqual({5: 'k'}, c)

    def test_resize_larger(self):
        if False:
            i = 10
            return i + 15
        c = fifo_cache.FIFOSizeCache(10, 8)
        c[1] = 'a'
        c[2] = 'bc'
        c[3] = 'def'
        c[4] = 'ghij'
        c.resize(12, 10)
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij'}, c)
        c[5] = 'kl'
        self.assertEqual({1: 'a', 2: 'bc', 3: 'def', 4: 'ghij', 5: 'kl'}, c)
        c[6] = 'mn'
        self.assertEqual({4: 'ghij', 5: 'kl', 6: 'mn'}, c)