"""Tests for list_comprehensions module."""
from nvidia.dali._autograph.converters import list_comprehensions
from nvidia.dali._autograph.core import converter_testing

class ListCompTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, f, *inputs):
        if False:
            print('Hello World!')
        tr = self.transform(f, list_comprehensions)
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_basic(self):
        if False:
            return 10

        def f(l):
            if False:
                while True:
                    i = 10
            s = [e * e for e in l]
            return s
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1, 2, 3])

    def test_multiple_generators(self):
        if False:
            while True:
                i = 10

        def f(l):
            if False:
                while True:
                    i = 10
            s = [e * e for sublist in l for e in sublist]
            return s
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [[1], [2], [3]])

    def test_cond(self):
        if False:
            while True:
                i = 10

        def f(l):
            if False:
                for i in range(10):
                    print('nop')
            s = [e * e for e in l if e > 1]
            return s
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1, 2, 3])