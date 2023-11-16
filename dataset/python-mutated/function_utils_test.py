"""Tests for Estimator related util."""
import functools
from tensorflow.python.platform import test
from tensorflow.python.util import function_utils

def silly_example_function():
    if False:
        print('Hello World!')
    pass

class SillyCallableClass(object):

    def __call__(self):
        if False:
            while True:
                i = 10
        pass

class FnArgsTest(test.TestCase):

    def test_simple_function(self):
        if False:
            return 10

        def fn(a, b):
            if False:
                print('Hello World!')
            return a + b
        self.assertEqual(('a', 'b'), function_utils.fn_args(fn))

    def test_callable(self):
        if False:
            while True:
                i = 10

        class Foo(object):

            def __call__(self, a, b):
                if False:
                    while True:
                        i = 10
                return a + b
        self.assertEqual(('a', 'b'), function_utils.fn_args(Foo()))

    def test_bound_method(self):
        if False:
            print('Hello World!')

        class Foo(object):

            def bar(self, a, b):
                if False:
                    i = 10
                    return i + 15
                return a + b
        self.assertEqual(('a', 'b'), function_utils.fn_args(Foo().bar))

    def test_bound_method_no_self(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(object):

            def bar(*args):
                if False:
                    print('Hello World!')
                return args[1] + args[2]
        self.assertEqual((), function_utils.fn_args(Foo().bar))

    def test_partial_function(self):
        if False:
            i = 10
            return i + 15
        expected_test_arg = 123

        def fn(a, test_arg):
            if False:
                for i in range(10):
                    print('nop')
            if test_arg != expected_test_arg:
                return ValueError('partial fn does not work correctly')
            return a
        wrapped_fn = functools.partial(fn, test_arg=123)
        self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))

    def test_partial_function_with_positional_args(self):
        if False:
            i = 10
            return i + 15
        expected_test_arg = 123

        def fn(test_arg, a):
            if False:
                while True:
                    i = 10
            if test_arg != expected_test_arg:
                return ValueError('partial fn does not work correctly')
            return a
        wrapped_fn = functools.partial(fn, 123)
        self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))
        self.assertEqual(3, wrapped_fn(3))
        self.assertEqual(3, wrapped_fn(a=3))

    def test_double_partial(self):
        if False:
            while True:
                i = 10
        expected_test_arg1 = 123
        expected_test_arg2 = 456

        def fn(a, test_arg1, test_arg2):
            if False:
                i = 10
                return i + 15
            if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
                return ValueError('partial does not work correctly')
            return a
        wrapped_fn = functools.partial(fn, test_arg2=456)
        double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)
        self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

    def test_double_partial_with_positional_args_in_outer_layer(self):
        if False:
            i = 10
            return i + 15
        expected_test_arg1 = 123
        expected_test_arg2 = 456

        def fn(test_arg1, a, test_arg2):
            if False:
                while True:
                    i = 10
            if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
                return ValueError('partial fn does not work correctly')
            return a
        wrapped_fn = functools.partial(fn, test_arg2=456)
        double_wrapped_fn = functools.partial(wrapped_fn, 123)
        self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))
        self.assertEqual(3, double_wrapped_fn(3))
        self.assertEqual(3, double_wrapped_fn(a=3))

    def test_double_partial_with_positional_args_in_both_layers(self):
        if False:
            print('Hello World!')
        expected_test_arg1 = 123
        expected_test_arg2 = 456

        def fn(test_arg1, test_arg2, a):
            if False:
                return 10
            if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
                return ValueError('partial fn does not work correctly')
            return a
        wrapped_fn = functools.partial(fn, 123)
        double_wrapped_fn = functools.partial(wrapped_fn, 456)
        self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))
        self.assertEqual(3, double_wrapped_fn(3))
        self.assertEqual(3, double_wrapped_fn(a=3))

class HasKwargsTest(test.TestCase):

    def test_simple_function(self):
        if False:
            return 10
        fn_has_kwargs = lambda **x: x
        self.assertTrue(function_utils.has_kwargs(fn_has_kwargs))
        fn_has_no_kwargs = lambda x: x
        self.assertFalse(function_utils.has_kwargs(fn_has_no_kwargs))

    def test_callable(self):
        if False:
            for i in range(10):
                print('nop')

        class FooHasKwargs(object):

            def __call__(self, **x):
                if False:
                    for i in range(10):
                        print('nop')
                del x
        self.assertTrue(function_utils.has_kwargs(FooHasKwargs()))

        class FooHasNoKwargs(object):

            def __call__(self, x):
                if False:
                    print('Hello World!')
                del x
        self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs()))

    def test_bound_method(self):
        if False:
            return 10

        class FooHasKwargs(object):

            def fn(self, **x):
                if False:
                    print('Hello World!')
                del x
        self.assertTrue(function_utils.has_kwargs(FooHasKwargs().fn))

        class FooHasNoKwargs(object):

            def fn(self, x):
                if False:
                    i = 10
                    return i + 15
                del x
        self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs().fn))

    def test_partial_function(self):
        if False:
            print('Hello World!')
        expected_test_arg = 123

        def fn_has_kwargs(test_arg, **x):
            if False:
                print('Hello World!')
            if test_arg != expected_test_arg:
                return ValueError('partial fn does not work correctly')
            return x
        wrapped_fn = functools.partial(fn_has_kwargs, test_arg=123)
        self.assertTrue(function_utils.has_kwargs(wrapped_fn))
        some_kwargs = dict(x=1, y=2, z=3)
        self.assertEqual(wrapped_fn(**some_kwargs), some_kwargs)

        def fn_has_no_kwargs(x, test_arg):
            if False:
                return 10
            if test_arg != expected_test_arg:
                return ValueError('partial fn does not work correctly')
            return x
        wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg=123)
        self.assertFalse(function_utils.has_kwargs(wrapped_fn))
        some_arg = 1
        self.assertEqual(wrapped_fn(some_arg), some_arg)

    def test_double_partial(self):
        if False:
            i = 10
            return i + 15
        expected_test_arg1 = 123
        expected_test_arg2 = 456

        def fn_has_kwargs(test_arg1, test_arg2, **x):
            if False:
                print('Hello World!')
            if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
                return ValueError('partial does not work correctly')
            return x
        wrapped_fn = functools.partial(fn_has_kwargs, test_arg2=456)
        double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)
        self.assertTrue(function_utils.has_kwargs(double_wrapped_fn))
        some_kwargs = dict(x=1, y=2, z=3)
        self.assertEqual(double_wrapped_fn(**some_kwargs), some_kwargs)

        def fn_has_no_kwargs(x, test_arg1, test_arg2):
            if False:
                print('Hello World!')
            if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
                return ValueError('partial does not work correctly')
            return x
        wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg2=456)
        double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)
        self.assertFalse(function_utils.has_kwargs(double_wrapped_fn))
        some_arg = 1
        self.assertEqual(double_wrapped_fn(some_arg), some_arg)

    def test_raises_type_error(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'should be a callable'):
            function_utils.has_kwargs('not a function')

class GetFuncNameTest(test.TestCase):

    def testWithSimpleFunction(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('silly_example_function', function_utils.get_func_name(silly_example_function))

    def testWithClassMethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('GetFuncNameTest.testWithClassMethod', function_utils.get_func_name(self.testWithClassMethod))

    def testWithCallableClass(self):
        if False:
            i = 10
            return i + 15
        callable_instance = SillyCallableClass()
        self.assertRegex(function_utils.get_func_name(callable_instance), '<.*SillyCallableClass.*>')

    def testWithFunctoolsPartial(self):
        if False:
            i = 10
            return i + 15
        partial = functools.partial(silly_example_function)
        self.assertRegex(function_utils.get_func_name(partial), '<.*functools.partial.*>')

    def testWithLambda(self):
        if False:
            print('Hello World!')
        anon_fn = lambda x: x
        self.assertEqual('<lambda>', function_utils.get_func_name(anon_fn))

    def testRaisesWithNonCallableObject(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            function_utils.get_func_name(None)

class GetFuncCodeTest(test.TestCase):

    def testWithSimpleFunction(self):
        if False:
            print('Hello World!')
        code = function_utils.get_func_code(silly_example_function)
        self.assertIsNotNone(code)
        self.assertRegex(code.co_filename, 'function_utils_test.py')

    def testWithClassMethod(self):
        if False:
            return 10
        code = function_utils.get_func_code(self.testWithClassMethod)
        self.assertIsNotNone(code)
        self.assertRegex(code.co_filename, 'function_utils_test.py')

    def testWithCallableClass(self):
        if False:
            while True:
                i = 10
        callable_instance = SillyCallableClass()
        code = function_utils.get_func_code(callable_instance)
        self.assertIsNotNone(code)
        self.assertRegex(code.co_filename, 'function_utils_test.py')

    def testWithLambda(self):
        if False:
            i = 10
            return i + 15
        anon_fn = lambda x: x
        code = function_utils.get_func_code(anon_fn)
        self.assertIsNotNone(code)
        self.assertRegex(code.co_filename, 'function_utils_test.py')

    def testWithFunctoolsPartial(self):
        if False:
            i = 10
            return i + 15
        partial = functools.partial(silly_example_function)
        code = function_utils.get_func_code(partial)
        self.assertIsNone(code)

    def testRaisesWithNonCallableObject(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            function_utils.get_func_code(None)
if __name__ == '__main__':
    test.main()