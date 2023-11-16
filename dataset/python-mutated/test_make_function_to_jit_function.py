from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest

@njit
def consumer(func, *args):
    if False:
        for i in range(10):
            print('nop')
    return func(*args)

@njit
def consumer2arg(func1, func2):
    if False:
        print('Hello World!')
    return func2(func1)
_global = 123

class TestMakeFunctionToJITFunction(unittest.TestCase):
    """
    This tests the pass that converts ir.Expr.op == make_function (i.e. closure)
    into a JIT function.
    """

    def test_escape(self):
        if False:
            print('Hello World!')

        def impl_factory(consumer_func):
            if False:
                print('Hello World!')

            def impl():
                if False:
                    for i in range(10):
                        print('nop')

                def inner():
                    if False:
                        return 10
                    return 10
                return consumer_func(inner)
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        self.assertEqual(impl(), cfunc())

    def test_nested_escape(self):
        if False:
            return 10

        def impl_factory(consumer_func):
            if False:
                print('Hello World!')

            def impl():
                if False:
                    i = 10
                    return i + 15

                def inner():
                    if False:
                        while True:
                            i = 10
                    return 10

                def innerinner(x):
                    if False:
                        print('Hello World!')
                    return x()
                return consumer_func(inner, innerinner)
            return impl
        cfunc = njit(impl_factory(consumer2arg))
        impl = impl_factory(consumer2arg.py_func)
        self.assertEqual(impl(), cfunc())

    def test_closure_in_escaper(self):
        if False:
            while True:
                i = 10

        def impl_factory(consumer_func):
            if False:
                while True:
                    i = 10

            def impl():
                if False:
                    for i in range(10):
                        print('nop')

                def callinner():
                    if False:
                        return 10

                    def inner():
                        if False:
                            i = 10
                            return i + 15
                        return 10
                    return inner()
                return consumer_func(callinner)
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        self.assertEqual(impl(), cfunc())

    def test_close_over_consts(self):
        if False:
            print('Hello World!')

        def impl_factory(consumer_func):
            if False:
                print('Hello World!')

            def impl():
                if False:
                    for i in range(10):
                        print('nop')
                y = 10

                def callinner(z):
                    if False:
                        return 10
                    return y + z + _global
                return consumer_func(callinner, 6)
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        self.assertEqual(impl(), cfunc())

    def test_close_over_consts_w_args(self):
        if False:
            for i in range(10):
                print('nop')

        def impl_factory(consumer_func):
            if False:
                while True:
                    i = 10

            def impl(x):
                if False:
                    print('Hello World!')
                y = 10

                def callinner(z):
                    if False:
                        print('Hello World!')
                    return y + z + _global
                return consumer_func(callinner, x)
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_with_overload(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(func, *args):
            if False:
                i = 10
                return i + 15
            nargs = len(args)
            if nargs == 1:
                return func(*args)
            elif nargs == 2:
                return func(func(*args))

        @overload(foo)
        def foo_ol(func, *args):
            if False:
                i = 10
                return i + 15
            nargs = len(args)
            if nargs == 1:

                def impl(func, *args):
                    if False:
                        return 10
                    return func(*args)
                return impl
            elif nargs == 2:

                def impl(func, *args):
                    if False:
                        i = 10
                        return i + 15
                    return func(func(*args))
                return impl

        def impl_factory(consumer_func):
            if False:
                return 10

            def impl(x):
                if False:
                    for i in range(10):
                        print('nop')
                y = 10

                def callinner(*z):
                    if False:
                        for i in range(10):
                            print('nop')
                    return y + np.sum(np.asarray(z)) + _global
                return (foo(callinner, x), foo(callinner, x, x))
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        a = 5
        self.assertEqual(impl(a), cfunc(a))

    def test_basic_apply_like_case(self):
        if False:
            for i in range(10):
                print('nop')

        def apply(array, func):
            if False:
                for i in range(10):
                    print('nop')
            return func(array)

        @overload(apply)
        def ov_apply(array, func):
            if False:
                return 10
            return lambda array, func: func(array)

        def impl(array):
            if False:
                i = 10
                return i + 15

            def mul10(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x * 10
            return apply(array, mul10)
        cfunc = njit(impl)
        a = np.arange(10)
        np.testing.assert_allclose(impl(a), cfunc(a))

    @unittest.skip('Needs option/flag inheritance to work')
    def test_jit_option_inheritance(self):
        if False:
            while True:
                i = 10

        def impl_factory(consumer_func):
            if False:
                print('Hello World!')

            def impl(x):
                if False:
                    print('Hello World!')

                def inner(val):
                    if False:
                        for i in range(10):
                            print('nop')
                    return 1 / val
                return consumer_func(inner, x)
            return impl
        cfunc = njit(error_model='numpy')(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        a = 0
        self.assertEqual(impl(a), cfunc(a))

    def test_multiply_defined_freevar(self):
        if False:
            return 10

        @njit
        def impl(c):
            if False:
                return 10
            if c:
                x = 3

                def inner(y):
                    if False:
                        print('Hello World!')
                    return y + x
                r = consumer(inner, 1)
            else:
                x = 6

                def inner(y):
                    if False:
                        print('Hello World!')
                    return y + x
                r = consumer(inner, 2)
            return r
        with self.assertRaises(errors.TypingError) as e:
            impl(1)
        self.assertIn('Cannot capture a constant value for variable', str(e.exception))

    def test_non_const_in_escapee(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def impl(x):
            if False:
                i = 10
                return i + 15
            z = np.arange(x)

            def inner(val):
                if False:
                    while True:
                        i = 10
                return 1 + z + val
            return consumer(inner, x)
        with self.assertRaises(errors.TypingError) as e:
            impl(1)
        self.assertIn('Cannot capture the non-constant value associated', str(e.exception))

    def test_escape_with_kwargs(self):
        if False:
            print('Hello World!')

        def impl_factory(consumer_func):
            if False:
                for i in range(10):
                    print('nop')

            def impl():
                if False:
                    i = 10
                    return i + 15
                t = 12

                def inner(a, b, c, mydefault1=123, mydefault2=456):
                    if False:
                        for i in range(10):
                            print('nop')
                    z = 4
                    return mydefault1 + mydefault2 + z + t + a + b + c
                return (inner(1, 2, 5, 91, 53), consumer_func(inner, 1, 2, 3, 73), consumer_func(inner, 1, 2, 3), inner(1, 2, 4))
            return impl
        cfunc = njit(impl_factory(consumer))
        impl = impl_factory(consumer.py_func)
        np.testing.assert_allclose(impl(), cfunc())

    def test_escape_with_kwargs_override_kwargs(self):
        if False:
            return 10

        @njit
        def specialised_consumer(func, *args):
            if False:
                return 10
            (x, y, z) = args
            a = func(x, y, z, mydefault1=1000)
            b = func(x, y, z, mydefault2=1000)
            c = func(x, y, z, mydefault1=1000, mydefault2=1000)
            return a + b + c

        def impl_factory(consumer_func):
            if False:
                i = 10
                return i + 15

            def impl():
                if False:
                    return 10
                t = 12

                def inner(a, b, c, mydefault1=123, mydefault2=456):
                    if False:
                        return 10
                    z = 4
                    return mydefault1 + mydefault2 + z + t + a + b + c
                return (inner(1, 2, 5, 91, 53), consumer_func(inner, 1, 2, 11), consumer_func(inner, 1, 2, 3), inner(1, 2, 4))
            return impl
        cfunc = njit(impl_factory(specialised_consumer))
        impl = impl_factory(specialised_consumer.py_func)
        np.testing.assert_allclose(impl(), cfunc())
if __name__ == '__main__':
    unittest.main()