"""Unit tests for object_identity."""
from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity

class ObjectIdentityWrapperTest(test.TestCase):

    def testWrapperNotEqualToWrapped(self):
        if False:
            for i in range(10):
                print('nop')

        class SettableHash(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.hash_value = 8675309

            def __hash__(self):
                if False:
                    return 10
                return self.hash_value
        o = SettableHash()
        wrap1 = object_identity._ObjectIdentityWrapper(o)
        wrap2 = object_identity._ObjectIdentityWrapper(o)
        self.assertEqual(wrap1, wrap1)
        self.assertEqual(wrap1, wrap2)
        self.assertEqual(o, wrap1.unwrapped)
        self.assertEqual(o, wrap2.unwrapped)
        with self.assertRaises(TypeError):
            bool(o == wrap1)
        with self.assertRaises(TypeError):
            bool(wrap1 != o)
        self.assertNotIn(o, set([wrap1]))
        o.hash_value = id(o)
        with self.assertRaises(TypeError):
            bool(o in set([wrap1]))

    def testNestFlatten(self):
        if False:
            return 10
        a = object_identity._ObjectIdentityWrapper('a')
        b = object_identity._ObjectIdentityWrapper('b')
        c = object_identity._ObjectIdentityWrapper('c')
        flat = nest.flatten([[[(a, b)]], c])
        self.assertEqual(flat, [a, b, c])

    def testNestMapStructure(self):
        if False:
            return 10
        k = object_identity._ObjectIdentityWrapper('k')
        v1 = object_identity._ObjectIdentityWrapper('v1')
        v2 = object_identity._ObjectIdentityWrapper('v2')
        struct = nest.map_structure(lambda a, b: (a, b), {k: v1}, {k: v2})
        self.assertEqual(struct, {k: (v1, v2)})

class ObjectIdentitySetTest(test.TestCase):

    def testDifference(self):
        if False:
            i = 10
            return i + 15

        class Element(object):
            pass
        a = Element()
        b = Element()
        c = Element()
        set1 = object_identity.ObjectIdentitySet([a, b])
        set2 = object_identity.ObjectIdentitySet([b, c])
        diff_set = set1.difference(set2)
        self.assertIn(a, diff_set)
        self.assertNotIn(b, diff_set)
        self.assertNotIn(c, diff_set)

    def testDiscard(self):
        if False:
            return 10
        a = object()
        b = object()
        set1 = object_identity.ObjectIdentitySet([a, b])
        set1.discard(a)
        self.assertIn(b, set1)
        self.assertNotIn(a, set1)

    def testClear(self):
        if False:
            while True:
                i = 10
        a = object()
        b = object()
        set1 = object_identity.ObjectIdentitySet([a, b])
        set1.clear()
        self.assertLen(set1, 0)
if __name__ == '__main__':
    test.main()