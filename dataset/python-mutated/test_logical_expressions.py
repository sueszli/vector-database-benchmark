"""Tests for logical_expressions module."""
from nvidia.dali._autograph.converters import logical_expressions
from nvidia.dali._autograph.core import converter_testing

class LogicalExpressionTest(converter_testing.TestCase):

    def test_equals(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            return a == b
        tr = self.transform(f, logical_expressions)
        self.assertTrue(tr(1, 1))
        self.assertFalse(tr(1, 2))

    def test_bool_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return (a or b) and (a or b or c) and (not c)
        tr = self.transform(f, logical_expressions)
        self.assertTrue(tr(True, False, False))
        self.assertFalse(tr(True, False, True))

    def test_comparison(self):
        if False:
            return 10

        def f(a, b, c, d):
            if False:
                print('Hello World!')
            return a < b == c > d
        tr = self.transform(f, logical_expressions)
        self.assertTrue(tr(1, 2, 2, 1))
        self.assertFalse(tr(1, 2, 2, 3))

    def test_default_ops(self):
        if False:
            return 10

        def f(a, b):
            if False:
                i = 10
                return i + 15
            return a in b
        tr = self.transform(f, logical_expressions)
        self.assertTrue(tr('a', ('a',)))

    def test_unary_ops(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                return 10
            return (~a, -a, +a)
        tr = self.transform(f, logical_expressions)
        self.assertEqual(tr(1), (-2, -1, 1))