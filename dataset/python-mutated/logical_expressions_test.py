"""Tests for logical_expressions module."""
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class LogicalExpressionTest(converter_testing.TestCase):

    def test_equals(self):
        if False:
            return 10

        def f(a, b):
            if False:
                return 10
            return a == b
        tr = self.transform(f, logical_expressions)
        self.assertTrue(self.evaluate(tr(constant_op.constant(1), 1)))
        self.assertFalse(self.evaluate(tr(constant_op.constant(1), 2)))

    @test_util.run_deprecated_v1
    def test_bool_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                return 10
            return (a or b) and (a or b or c) and (not c)
        tr = self.transform(f, logical_expressions)
        self.assertTrue(self.evaluate(tr(constant_op.constant(True), False, False)))
        self.assertFalse(self.evaluate(tr(constant_op.constant(True), False, True)))

    def test_comparison(self):
        if False:
            return 10

        def f(a, b, c, d):
            if False:
                while True:
                    i = 10
            return a < b == c > d
        tr = self.transform(f, logical_expressions)
        self.assertTrue(self.evaluate(tr(constant_op.constant(1), 2, 2, 1)))
        self.assertFalse(self.evaluate(tr(constant_op.constant(1), 2, 2, 3)))

    def test_default_ops(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                print('Hello World!')
            return a in b
        tr = self.transform(f, logical_expressions)
        self.assertTrue(tr('a', ('a',)))

    def test_unary_ops(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            return (~a, -a, +a)
        tr = self.transform(f, logical_expressions)
        self.assertEqual(tr(1), (-2, -1, 1))
if __name__ == '__main__':
    test.main()