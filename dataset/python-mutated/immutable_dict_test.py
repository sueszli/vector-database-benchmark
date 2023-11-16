"""Tests for tf.framework.immutable_dict."""
from absl.testing import parameterized
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class ImmutableDictTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    def testGetItem(self):
        if False:
            i = 10
            return i + 15
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(d['x'], 1)
        self.assertEqual(d['y'], 2)
        with self.assertRaises(KeyError):
            d['z']

    def testIter(self):
        if False:
            return 10
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(set(iter(d)), set(['x', 'y']))

    def testContains(self):
        if False:
            for i in range(10):
                print('nop')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertIn('x', d)
        self.assertIn('y', d)
        self.assertNotIn('z', d)

    def testLen(self):
        if False:
            i = 10
            return i + 15
        d1 = immutable_dict.ImmutableDict({})
        self.assertLen(d1, 0)
        d2 = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertLen(d2, 2)

    def testRepr(self):
        if False:
            return 10
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        s = repr(d)
        self.assertTrue(s == "ImmutableDict({'x': 1, 'y': 2})" or s == "ImmutableDict({'y': 1, 'x': 2})")

    def testGet(self):
        if False:
            while True:
                i = 10
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(d.get('x'), 1)
        self.assertEqual(d.get('y'), 2)
        self.assertIsNone(d.get('z'))
        self.assertEqual(d.get('z', 'Foo'), 'Foo')

    def testKeys(self):
        if False:
            return 10
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(set(d.keys()), set(['x', 'y']))

    def testValues(self):
        if False:
            print('Hello World!')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(set(d.values()), set([1, 2]))

    def testItems(self):
        if False:
            i = 10
            return i + 15
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(set(d.items()), set([('x', 1), ('y', 2)]))

    def testEqual(self):
        if False:
            for i in range(10):
                print('nop')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertEqual(d, {'x': 1, 'y': 2})

    def testNotEqual(self):
        if False:
            for i in range(10):
                print('nop')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        self.assertNotEqual(d, {'x': 1})

    def testSetItemFails(self):
        if False:
            for i in range(10):
                print('nop')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        with self.assertRaises(TypeError):
            d['x'] = 5
        with self.assertRaises(TypeError):
            d['z'] = 5

    def testDelItemFails(self):
        if False:
            print('Hello World!')
        d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
        with self.assertRaises(TypeError):
            del d['x']
        with self.assertRaises(TypeError):
            del d['z']
if __name__ == '__main__':
    googletest.main()