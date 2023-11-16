"""Tests for continue_statements module."""
from nvidia.dali._autograph.converters import continue_statements
from nvidia.dali._autograph.core import converter_testing

class ContinueCanonicalizationTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, f, *inputs):
        if False:
            for i in range(10):
                print('nop')
        tr = self.transform(f, continue_statements)
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_basic(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                while True:
                    i = 10
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_multiple_continues(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            v = []
            while x > 0:
                x -= 1
                if x > 1:
                    continue
                if x > 2:
                    continue
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_multiple_continues_in_nested_scope(self):
        if False:
            return 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            v = []
            for x in a:
                x -= 1
                if x > 100:
                    continue
                try:
                    raise ValueError('intentional')
                except ValueError:
                    continue
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1])
        self.assertTransformedEquivalent(f, [2])
        self.assertTransformedEquivalent(f, [1, 2, 3])

    def test_for_loop(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                print('Hello World!')
            v = []
            for x in a:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1])
        self.assertTransformedEquivalent(f, [2])
        self.assertTransformedEquivalent(f, [1, 2, 3])

    def test_nested_with(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_nested_multiple_withs(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_nested_multiple_withs_and_statements(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
                v.append(x)
                v.append(x)
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_nested_multiple_withs_and_nested_withs(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    continue
                v.append(x)
                v.append(x)
                v.append(x)
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_nested(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            v = []
            u = []
            w = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    if x % 3 != 0:
                        u.append(x)
                    else:
                        w.append(x)
                        continue
                v.append(x)
            return (v, u, w)
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_multiple_guarded_continues_with_side_effects(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')

            def track(u, x):
                if False:
                    i = 10
                    return i + 15
                u.append(x)
                return x
            u = []
            v = []
            while x > 0:
                x -= 1
                if track(u, x) > 1:
                    continue
                if track(u, x) > 2:
                    continue
                v.append(x)
            return (u, v)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 2)