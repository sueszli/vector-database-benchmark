"""Tests for mixin.py."""
from pytype.abstract import mixin
import unittest

class MixinMetaTest(unittest.TestCase):

    def test_mixin_super(self):
        if False:
            for i in range(10):
                print('nop')
        "Test the imitation 'super' method on MixinMeta."

        class A:

            def f(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class MyMixin(metaclass=mixin.MixinMeta):
            overloads = ('f',)

            def f(self, x):
                if False:
                    return 10
                if x == 0:
                    return 'hello'
                return MyMixin.super(self.f)(x)

        class B(A, MyMixin):
            pass
        b = B()
        v_mixin = b.f(0)
        v_a = b.f(1)
        self.assertEqual(v_mixin, 'hello')
        self.assertEqual(v_a, 1)

class PythonDictTest(unittest.TestCase):

    def test_wraps_dict(self):
        if False:
            print('Hello World!')

        class A(mixin.PythonDict):

            def __init__(self, pyval):
                if False:
                    print('Hello World!')
                mixin.PythonDict.init_mixin(self, pyval)
        a = A({'foo': 1, 'bar': 2})
        self.assertEqual(a.get('x', 'baz'), 'baz')
        self.assertNotIn('x', a)
        self.assertEqual(a.get('foo'), 1)
        self.assertEqual(a['foo'], 1)
        self.assertIn('foo', a)
        self.assertIn('bar', a)
        self.assertEqual(a.copy(), a.pyval)
        self.assertCountEqual(iter(a), ['foo', 'bar'])
        self.assertCountEqual(a.keys(), ['foo', 'bar'])
        self.assertCountEqual(a.values(), [1, 2])
        self.assertCountEqual(a.items(), [('foo', 1), ('bar', 2)])
if __name__ == '__main__':
    unittest.main()