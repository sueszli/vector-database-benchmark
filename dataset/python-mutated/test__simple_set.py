"""Tests for the StaticTupleInterned type."""
import sys
from bzrlib import tests
from bzrlib.tests import features
try:
    from bzrlib import _simple_set_pyx
except ImportError:
    _simple_set_pyx = None

class _Hashable(object):
    """A simple object which has a fixed hash value.

    We could have used an 'int', but it turns out that Int objects don't
    implement tp_richcompare...
    """

    def __init__(self, the_hash):
        if False:
            for i in range(10):
                print('nop')
        self.hash = the_hash

    def __hash__(self):
        if False:
            return 10
        return self.hash

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, _Hashable):
            return NotImplemented
        return other.hash == self.hash

class _BadSecondHash(_Hashable):

    def __init__(self, the_hash):
        if False:
            while True:
                i = 10
        _Hashable.__init__(self, the_hash)
        self._first = True

    def __hash__(self):
        if False:
            while True:
                i = 10
        if self._first:
            self._first = False
            return self.hash
        else:
            raise ValueError('I can only be hashed once.')

class _BadCompare(_Hashable):

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('I refuse to play nice')

class _NoImplementCompare(_Hashable):

    def __eq__(self, other):
        if False:
            return 10
        return NotImplemented
compiled_simpleset_feature = features.ModuleAvailableFeature('bzrlib._simple_set_pyx')

class TestSimpleSet(tests.TestCase):
    _test_needs_features = [compiled_simpleset_feature]
    module = _simple_set_pyx

    def assertIn(self, obj, container):
        if False:
            i = 10
            return i + 15
        self.assertTrue(obj in container, '%s not found in %s' % (obj, container))

    def assertNotIn(self, obj, container):
        if False:
            i = 10
            return i + 15
        self.assertTrue(obj not in container, 'We found %s in %s' % (obj, container))

    def assertFillState(self, used, fill, mask, obj):
        if False:
            i = 10
            return i + 15
        self.assertEqual((used, fill, mask), (obj.used, obj.fill, obj.mask))

    def assertLookup(self, offset, value, obj, key):
        if False:
            while True:
                i = 10
        self.assertEqual((offset, value), obj._test_lookup(key))

    def assertRefcount(self, count, obj):
        if False:
            i = 10
            return i + 15
        'Assert that the refcount for obj is what we expect.\n\n        Note that this automatically adjusts for the fact that calling\n        assertRefcount actually creates a new pointer, as does calling\n        sys.getrefcount. So pass the expected value *before* the call.\n        '
        self.assertEqual(count, sys.getrefcount(obj) - 3)

    def test_initial(self):
        if False:
            i = 10
            return i + 15
        obj = self.module.SimpleSet()
        self.assertEqual(0, len(obj))
        st = ('foo', 'bar')
        self.assertFillState(0, 0, 1023, obj)

    def test__lookup(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        self.assertLookup(643, '<null>', obj, _Hashable(643))
        self.assertLookup(643, '<null>', obj, _Hashable(643 + 1024))
        self.assertLookup(643, '<null>', obj, _Hashable(643 + 50 * 1024))

    def test__lookup_collision(self):
        if False:
            while True:
                i = 10
        obj = self.module.SimpleSet()
        k1 = _Hashable(643)
        k2 = _Hashable(643 + 1024)
        self.assertLookup(643, '<null>', obj, k1)
        self.assertLookup(643, '<null>', obj, k2)
        obj.add(k1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, '<null>', obj, k2)

    def test__lookup_after_resize(self):
        if False:
            while True:
                i = 10
        obj = self.module.SimpleSet()
        k1 = _Hashable(643)
        k2 = _Hashable(643 + 1024)
        obj.add(k1)
        obj.add(k2)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)
        obj._py_resize(2047)
        self.assertEqual(2048, obj.mask + 1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(643 + 1024, k2, obj, k2)
        obj._py_resize(1023)
        self.assertEqual(1024, obj.mask + 1)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)

    def test_get_set_del_with_collisions(self):
        if False:
            i = 10
            return i + 15
        obj = self.module.SimpleSet()
        h1 = 643
        h2 = 643 + 1024
        h3 = 643 + 1024 * 50
        h4 = 643 + 1024 * 25
        h5 = 644
        h6 = 644 + 1024
        k1 = _Hashable(h1)
        k2 = _Hashable(h2)
        k3 = _Hashable(h3)
        k4 = _Hashable(h4)
        k5 = _Hashable(h5)
        k6 = _Hashable(h6)
        self.assertLookup(643, '<null>', obj, k1)
        self.assertLookup(643, '<null>', obj, k2)
        self.assertLookup(643, '<null>', obj, k3)
        self.assertLookup(643, '<null>', obj, k4)
        self.assertLookup(644, '<null>', obj, k5)
        self.assertLookup(644, '<null>', obj, k6)
        obj.add(k1)
        self.assertIn(k1, obj)
        self.assertNotIn(k2, obj)
        self.assertNotIn(k3, obj)
        self.assertNotIn(k4, obj)
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, '<null>', obj, k2)
        self.assertLookup(644, '<null>', obj, k3)
        self.assertLookup(644, '<null>', obj, k4)
        self.assertLookup(644, '<null>', obj, k5)
        self.assertLookup(644, '<null>', obj, k6)
        self.assertIs(k1, obj[k1])
        self.assertIs(k2, obj.add(k2))
        self.assertIs(k2, obj[k2])
        self.assertLookup(643, k1, obj, k1)
        self.assertLookup(644, k2, obj, k2)
        self.assertLookup(646, '<null>', obj, k3)
        self.assertLookup(646, '<null>', obj, k4)
        self.assertLookup(645, '<null>', obj, k5)
        self.assertLookup(645, '<null>', obj, k6)
        self.assertLookup(643, k1, obj, _Hashable(h1))
        self.assertLookup(644, k2, obj, _Hashable(h2))
        self.assertLookup(646, '<null>', obj, _Hashable(h3))
        self.assertLookup(646, '<null>', obj, _Hashable(h4))
        self.assertLookup(645, '<null>', obj, _Hashable(h5))
        self.assertLookup(645, '<null>', obj, _Hashable(h6))
        obj.add(k3)
        self.assertIs(k3, obj[k3])
        self.assertIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertIn(k3, obj)
        self.assertNotIn(k4, obj)
        obj.discard(k1)
        self.assertLookup(643, '<dummy>', obj, k1)
        self.assertLookup(644, k2, obj, k2)
        self.assertLookup(646, k3, obj, k3)
        self.assertLookup(643, '<dummy>', obj, k4)
        self.assertNotIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertIn(k3, obj)
        self.assertNotIn(k4, obj)

    def test_add(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        self.assertFillState(0, 0, 1023, obj)
        k1 = tuple(['foo'])
        self.assertRefcount(1, k1)
        self.assertIs(k1, obj.add(k1))
        self.assertFillState(1, 1, 1023, obj)
        self.assertRefcount(2, k1)
        ktest = obj[k1]
        self.assertRefcount(3, k1)
        self.assertIs(k1, ktest)
        del ktest
        self.assertRefcount(2, k1)
        k2 = tuple(['foo'])
        self.assertRefcount(1, k2)
        self.assertIsNot(k1, k2)
        self.assertIs(k1, obj.add(k2))
        self.assertFillState(1, 1, 1023, obj)
        self.assertRefcount(2, k1)
        self.assertRefcount(1, k2)
        self.assertIs(k1, obj[k1])
        self.assertIs(k1, obj[k2])
        self.assertRefcount(2, k1)
        self.assertRefcount(1, k2)
        obj.discard(k1)
        self.assertFillState(0, 1, 1023, obj)
        self.assertRefcount(1, k1)
        k3 = tuple(['bar'])
        self.assertRefcount(1, k3)
        self.assertIs(k3, obj.add(k3))
        self.assertFillState(1, 2, 1023, obj)
        self.assertRefcount(2, k3)
        self.assertIs(k2, obj.add(k2))
        self.assertFillState(2, 2, 1023, obj)
        self.assertRefcount(1, k1)
        self.assertRefcount(2, k2)
        self.assertRefcount(2, k3)

    def test_discard(self):
        if False:
            i = 10
            return i + 15
        obj = self.module.SimpleSet()
        k1 = tuple(['foo'])
        k2 = tuple(['foo'])
        k3 = tuple(['bar'])
        self.assertRefcount(1, k1)
        self.assertRefcount(1, k2)
        self.assertRefcount(1, k3)
        obj.add(k1)
        self.assertRefcount(2, k1)
        self.assertEqual(0, obj.discard(k3))
        self.assertRefcount(1, k3)
        obj.add(k3)
        self.assertRefcount(2, k3)
        self.assertEqual(1, obj.discard(k3))
        self.assertRefcount(1, k3)

    def test__resize(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.module.SimpleSet()
        k1 = ('foo',)
        k2 = ('bar',)
        k3 = ('baz',)
        obj.add(k1)
        obj.add(k2)
        obj.add(k3)
        obj.discard(k2)
        self.assertFillState(2, 3, 1023, obj)
        self.assertEqual(1024, obj._py_resize(500))
        self.assertFillState(2, 2, 1023, obj)
        obj.add(k2)
        obj.discard(k3)
        self.assertFillState(2, 3, 1023, obj)
        self.assertEqual(4096, obj._py_resize(4095))
        self.assertFillState(2, 2, 4095, obj)
        self.assertIn(k1, obj)
        self.assertIn(k2, obj)
        self.assertNotIn(k3, obj)
        obj.add(k2)
        self.assertIn(k2, obj)
        obj.discard(k2)
        self.assertEqual((591, '<dummy>'), obj._test_lookup(k2))
        self.assertFillState(1, 2, 4095, obj)
        self.assertEqual(2048, obj._py_resize(1024))
        self.assertFillState(1, 1, 2047, obj)
        self.assertEqual((591, '<null>'), obj._test_lookup(k2))

    def test_second_hash_failure(self):
        if False:
            i = 10
            return i + 15
        obj = self.module.SimpleSet()
        k1 = _BadSecondHash(200)
        k2 = _Hashable(200)
        obj.add(k1)
        self.assertFalse(k1._first)
        self.assertRaises(ValueError, obj.add, k2)

    def test_richcompare_failure(self):
        if False:
            return 10
        obj = self.module.SimpleSet()
        k1 = _Hashable(200)
        k2 = _BadCompare(200)
        obj.add(k1)
        self.assertRaises(RuntimeError, obj.add, k2)

    def test_richcompare_not_implemented(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        k1 = _NoImplementCompare(200)
        k2 = _NoImplementCompare(200)
        self.assertLookup(200, '<null>', obj, k1)
        self.assertLookup(200, '<null>', obj, k2)
        self.assertIs(k1, obj.add(k1))
        self.assertLookup(200, k1, obj, k1)
        self.assertLookup(201, '<null>', obj, k2)
        self.assertIs(k2, obj.add(k2))
        self.assertIs(k1, obj[k1])

    def test_add_and_remove_lots_of_items(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'
        for i in chars:
            for j in chars:
                k = (i, j)
                obj.add(k)
        num = len(chars) * len(chars)
        self.assertFillState(num, num, 8191, obj)
        for i in chars:
            for j in chars:
                k = (i, j)
                obj.discard(k)
        self.assertFillState(0, obj.fill, 1023, obj)
        self.assertTrue(obj.fill < 1024 / 5)

    def test__iter__(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        k1 = ('1',)
        k2 = ('1', '2')
        k3 = ('3', '4')
        obj.add(k1)
        obj.add(k2)
        obj.add(k3)
        all = set()
        for key in obj:
            all.add(key)
        self.assertEqual(sorted([k1, k2, k3]), sorted(all))
        iterator = iter(obj)
        iterator.next()
        obj.add(('foo',))
        self.assertRaises(RuntimeError, iterator.next)
        obj.discard(k2)
        self.assertRaises(RuntimeError, iterator.next)

    def test__sizeof__(self):
        if False:
            print('Hello World!')
        obj = self.module.SimpleSet()
        self.assertTrue(obj.__sizeof__() > 4096)