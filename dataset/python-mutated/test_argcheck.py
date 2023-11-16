from unittest import TestCase
from zipline.utils.argcheck import verify_callable_argspec, Argument, NoStarargs, UnexpectedStarargs, NoKwargs, UnexpectedKwargs, NotCallable, NotEnoughArguments, TooManyArguments, MismatchedArguments

class TestArgCheck(TestCase):

    def test_not_callable(self):
        if False:
            return 10
        '\n        Check the results of a non-callable object.\n        '
        not_callable = 'a'
        with self.assertRaises(NotCallable):
            verify_callable_argspec(not_callable)

    def test_no_starargs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests when a function does not have *args and it was expected.\n        '

        def f(a):
            if False:
                print('Hello World!')
            pass
        with self.assertRaises(NoStarargs):
            verify_callable_argspec(f, expect_starargs=True)

    def test_starargs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests when a function has *args and it was expected.\n        '

        def f(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass
        verify_callable_argspec(f, expect_starargs=True)

    def test_unexcpected_starargs(self):
        if False:
            while True:
                i = 10
        '\n        Tests a function that unexpectedly accepts *args.\n        '

        def f(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass
        with self.assertRaises(UnexpectedStarargs):
            verify_callable_argspec(f, expect_starargs=False)

    def test_ignore_starargs(self):
        if False:
            return 10
        '\n        Tests checking a function ignoring the presence of *args.\n        '

        def f(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def g():
            if False:
                while True:
                    i = 10
            pass
        verify_callable_argspec(f, expect_starargs=Argument.ignore)
        verify_callable_argspec(g, expect_starargs=Argument.ignore)

    def test_no_kwargs(self):
        if False:
            return 10
        '\n        Tests when a function does not have **kwargs and it was expected.\n        '

        def f():
            if False:
                for i in range(10):
                    print('nop')
            pass
        with self.assertRaises(NoKwargs):
            verify_callable_argspec(f, expect_kwargs=True)

    def test_kwargs(self):
        if False:
            while True:
                i = 10
        '\n        Tests when a function has **kwargs and it was expected.\n        '

        def f(**kwargs):
            if False:
                i = 10
                return i + 15
            pass
        verify_callable_argspec(f, expect_kwargs=True)

    def test_unexpected_kwargs(self):
        if False:
            print('Hello World!')
        '\n        Tests a function that unexpectedly accepts **kwargs.\n        '

        def f(**kwargs):
            if False:
                return 10
            pass
        with self.assertRaises(UnexpectedKwargs):
            verify_callable_argspec(f, expect_kwargs=False)

    def test_ignore_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests checking a function ignoring the presence of **kwargs.\n        '

        def f(**kwargs):
            if False:
                print('Hello World!')
            pass

        def g():
            if False:
                i = 10
                return i + 15
            pass
        verify_callable_argspec(f, expect_kwargs=Argument.ignore)
        verify_callable_argspec(g, expect_kwargs=Argument.ignore)

    def test_arg_subset(self):
        if False:
            return 10
        '\n        Tests when the args are a subset of the expectations.\n        '

        def f(a, b):
            if False:
                while True:
                    i = 10
            pass
        with self.assertRaises(NotEnoughArguments):
            verify_callable_argspec(f, [Argument('a'), Argument('b'), Argument('c')])

    def test_arg_superset(self):
        if False:
            return 10

        def f(a, b, c):
            if False:
                return 10
            pass
        with self.assertRaises(TooManyArguments):
            verify_callable_argspec(f, [Argument('a'), Argument('b')])

    def test_no_default(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests when an argument expects a default and it is not present.\n        '

        def f(a):
            if False:
                i = 10
                return i + 15
            pass
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('a', 1)])

    def test_default(self):
        if False:
            print('Hello World!')
        '\n        Tests when an argument expects a default and it is present.\n        '

        def f(a=1):
            if False:
                print('Hello World!')
            pass
        verify_callable_argspec(f, [Argument('a', 1)])

    def test_ignore_default(self):
        if False:
            return 10
        '\n        Tests that ignoring defaults works as intended.\n        '

        def f(a=1):
            if False:
                while True:
                    i = 10
            pass
        verify_callable_argspec(f, [Argument('a')])

    def test_mismatched_args(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                i = 10
                return i + 15
            pass
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('c'), Argument('d')])

    def test_ignore_args(self):
        if False:
            print('Hello World!')
        '\n        Tests the ignore argument list feature.\n        '

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def g():
            if False:
                i = 10
                return i + 15
            pass
        h = 'not_callable'
        verify_callable_argspec(f)
        verify_callable_argspec(g)
        with self.assertRaises(NotCallable):
            verify_callable_argspec(h)

    def test_out_of_order(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests the case where arguments are not in the correct order.\n        '

        def f(a, b):
            if False:
                while True:
                    i = 10
            pass
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('b'), Argument('a')])

    def test_wrong_default(self):
        if False:
            return 10
        '\n        Tests the case where a default is expected, but the default provided\n        does not match the one expected.\n        '

        def f(a=1):
            if False:
                return 10
            pass
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('a', 2)])

    def test_any_default(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests the any_default option.\n        '

        def f(a=1):
            if False:
                return 10
            pass

        def g(a=2):
            if False:
                print('Hello World!')
            pass

        def h(a):
            if False:
                while True:
                    i = 10
            pass
        expected_args = [Argument('a', Argument.any_default)]
        verify_callable_argspec(f, expected_args)
        verify_callable_argspec(g, expected_args)
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(h, expected_args)

    def test_ignore_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests ignoring a param name.\n        '

        def f(a):
            if False:
                while True:
                    i = 10
            pass

        def g(b):
            if False:
                while True:
                    i = 10
            pass

        def h(c=1):
            if False:
                print('Hello World!')
            pass
        expected_args = [Argument(Argument.ignore, Argument.no_default)]
        verify_callable_argspec(f, expected_args)
        verify_callable_argspec(f, expected_args)
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(h, expected_args)

    def test_bound_method(self):
        if False:
            i = 10
            return i + 15

        class C(object):

            def f(self, a, b):
                if False:
                    return 10
                pass
        method = C().f
        verify_callable_argspec(method, [Argument('a'), Argument('b')])
        with self.assertRaises(NotEnoughArguments):
            verify_callable_argspec(method, [Argument('self'), Argument('a'), Argument('b')])