"""Tests for return_statements module."""
from nvidia.dali._autograph.converters import functions
from nvidia.dali._autograph.converters import return_statements
from nvidia.dali._autograph.core import converter_testing

class SingleReturnTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, f, *inputs):
        if False:
            return 10
        tr = self.transform(f, (functions, return_statements))
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_straightline(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        self.assertTransformedEquivalent(f, 2)

    def test_superfluous_returns(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                return 10
            retval = 1
            return retval
            retval = 2
            return retval
        self.assertTransformedEquivalent(f)

    def test_superfluous_returns_adjacent(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return 1
            return 2
        self.assertTransformedEquivalent(f)

    def test_conditional(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            if x > 0:
                return x
            else:
                return x * x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_conditional_missing_else(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            if x > 0:
                return x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_conditional_missing_else_then_default(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                print('Hello World!')
            if x > 0:
                return x
            return x * x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_conditional_else_only_then_default(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            if x < 0:
                x *= x
            else:
                return x
            return x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_conditional_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            if x > 0:
                if x < 5:
                    return x
                else:
                    return x * x
            else:
                return x * x * x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)
        self.assertTransformedEquivalent(f, 5)

    def test_context_manager(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x * x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_context_manager_in_conditional(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            if x > 0:
                return x * x
            else:
                return x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def text_conditional_in_context_manager(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            if x > 0:
                return x * x
            else:
                return x
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_no_return(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                print('Hello World!')
            x *= x
        self.assertTransformedEquivalent(f, 2)

    def test_nested_function(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10

            def inner_fn(y):
                if False:
                    return 10
                if y > 0:
                    return y * y
                else:
                    return y
            return inner_fn(x)
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_nested_function_in_control_flow(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                while True:
                    i = 10
            if x:

                def inner_fn(y):
                    if False:
                        i = 10
                        return i + 15
                    return y
                inner_fn(x)
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, -2)

    def test_for_loop(self):
        if False:
            return 10

        def f(n):
            if False:
                print('Hello World!')
            for _ in range(n):
                return 1
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, 0)

    def test_while_loop(self):
        if False:
            while True:
                i = 10

        def f(n):
            if False:
                while True:
                    i = 10
            i = 0
            s = 0
            while i < n:
                i += 1
                s += i
                if s > 4:
                    return s
            return -1
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, 4)

    def test_null_return(self):
        if False:
            print('Hello World!')

        def f(n):
            if False:
                return 10
            if n > 4:
                return
            return
        self.assertTransformedEquivalent(f, 4)
        self.assertTransformedEquivalent(f, 5)

    def test_nested_multiple_withs(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    return v
                v.append(x)
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 4)

    def test_multiple_returns_in_nested_scope(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                while True:
                    i = 10
            v = []
            for x in a:
                x -= 1
                if x > 100:
                    return v
                try:
                    raise ValueError('intentional')
                except ValueError:
                    return v
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1])
        self.assertTransformedEquivalent(f, [2])
        self.assertTransformedEquivalent(f, [1, 2, 3])