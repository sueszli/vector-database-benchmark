"""Tests for conditional_expressions module."""
from nvidia.dali._autograph.converters import conditional_expressions
from nvidia.dali._autograph.core import converter_testing

class ConditionalExpressionsTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, f, *inputs):
        if False:
            i = 10
            return i + 15
        tr = self.transform(f, conditional_expressions)
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_basic(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            return 1 if x else 0
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 3)

    def test_nested_orelse(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                i = 10
                return i + 15
            y = x * x if x > 0 else x if x else 1
            return y
        self.assertTransformedEquivalent(f, -2)
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 2)