from twisted.trial import unittest
from buildbot import util

class ComparableMixin(unittest.TestCase):

    class Foo(util.ComparableMixin):
        compare_attrs = ('a', 'b')

        def __init__(self, a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            (self.a, self.b, self.c) = (a, b, c)

    class Bar(Foo, util.ComparableMixin):
        compare_attrs = ('b', 'c')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.f123 = self.Foo(1, 2, 3)
        self.f124 = self.Foo(1, 2, 4)
        self.f134 = self.Foo(1, 3, 4)
        self.b123 = self.Bar(1, 2, 3)
        self.b223 = self.Bar(2, 2, 3)
        self.b213 = self.Bar(2, 1, 3)

    def test_equality_identity(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.f123, self.f123)

    def test_equality_same(self):
        if False:
            print('Hello World!')
        another_f123 = self.Foo(1, 2, 3)
        self.assertEqual(self.f123, another_f123)

    def test_equality_unimportantDifferences(self):
        if False:
            return 10
        self.assertEqual(self.f123, self.f124)

    def test_inequality_unimportantDifferences_subclass(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(self.b123, self.b223)

    def test_inequality_importantDifferences(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(self.f123, self.f134)

    def test_inequality_importantDifferences_subclass(self):
        if False:
            return 10
        self.assertNotEqual(self.b123, self.b213)

    def test_inequality_differentClasses(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(self.f123, self.b123)

    def test_instance_attribute_not_used(self):
        if False:
            while True:
                i = 10
        another_f123 = self.Foo(1, 2, 3)
        another_f123.compare_attrs = ('b', 'a')
        self.assertEqual(self.f123, another_f123)

    def test_ne_importantDifferences(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(self.f123, self.f134)

    def test_ne_differentClasses(self):
        if False:
            return 10
        self.assertNotEqual(self.f123, self.b123)

    def test_compare(self):
        if False:
            return 10
        self.assertEqual(self.f123, self.f123)
        self.assertNotEqual(self.b223, self.b213)
        self.assertGreater(self.b223, self.b213)
        self.assertFalse(self.b223 > self.f123)
        self.assertGreaterEqual(self.b223, self.b213)
        self.assertGreaterEqual(self.b223, self.b223)
        self.assertFalse(self.f123 >= self.b123)
        self.assertLess(self.b213, self.b223)
        self.assertLessEqual(self.b213, self.b223)
        self.assertLessEqual(self.b213, self.b213)
        self.assertFalse(self.f123 <= self.b123)