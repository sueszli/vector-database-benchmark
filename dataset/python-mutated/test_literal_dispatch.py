import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import overload, SentryLiteralArgs, overload_method, register_model, intrinsic
from numba.misc.special import literally

class TestLiteralDispatch(TestCase):

    def check_literal_basic(self, literal_args):
        if False:
            print('Hello World!')

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return literally(x)
        for lit in literal_args:
            self.assertEqual(foo(lit), lit)
        for (lit, sig) in zip(literal_args, foo.signatures):
            self.assertEqual(sig[0].literal_value, lit)

    def test_literal_basic(self):
        if False:
            while True:
                i = 10
        self.check_literal_basic([123, 321])
        self.check_literal_basic(['abc', 'cb123'])

    def test_literal_nested(self):
        if False:
            print('Hello World!')

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return literally(x) * 2

        @njit
        def bar(y, x):
            if False:
                for i in range(10):
                    print('nop')
            return foo(y) + x
        (y, x) = (3, 7)
        self.assertEqual(bar(y, x), y * 2 + x)
        [foo_sig] = foo.signatures
        self.assertEqual(foo_sig[0], types.literal(y))
        [bar_sig] = bar.signatures
        self.assertEqual(bar_sig[0], types.literal(y))
        self.assertNotIsInstance(bar_sig[1], types.Literal)

    def test_literally_freevar(self):
        if False:
            while True:
                i = 10
        import numba

        @njit
        def foo(x):
            if False:
                print('Hello World!')
            return numba.literally(x)
        self.assertEqual(foo(123), 123)
        self.assertEqual(foo.signatures[0][0], types.literal(123))

    def test_mutual_recursion_literal(self):
        if False:
            return 10

        def get_functions(decor):
            if False:
                for i in range(10):
                    print('nop')

            @decor
            def outer_fac(n, value):
                if False:
                    for i in range(10):
                        print('nop')
                if n < 1:
                    return value
                return n * inner_fac(n - 1, value)

            @decor
            def inner_fac(n, value):
                if False:
                    while True:
                        i = 10
                if n < 1:
                    return literally(value)
                return n * outer_fac(n - 1, value)
            return (outer_fac, inner_fac)
        (ref_outer_fac, ref_inner_fac) = get_functions(lambda x: x)
        (outer_fac, inner_fac) = get_functions(njit)
        self.assertEqual(outer_fac(10, 12), ref_outer_fac(10, 12))
        self.assertEqual(outer_fac.signatures[0][1].literal_value, 12)
        self.assertEqual(inner_fac.signatures[0][1].literal_value, 12)
        self.assertEqual(inner_fac(11, 13), ref_inner_fac(11, 13))
        self.assertEqual(outer_fac.signatures[1][1].literal_value, 13)
        self.assertEqual(inner_fac.signatures[1][1].literal_value, 13)

    def test_literal_nested_multi_arg(self):
        if False:
            return 10

        @njit
        def foo(a, b, c):
            if False:
                print('Hello World!')
            return inner(a, c)

        @njit
        def inner(x, y):
            if False:
                print('Hello World!')
            return x + literally(y)
        kwargs = dict(a=1, b=2, c=3)
        got = foo(**kwargs)
        expect = (lambda a, b, c: a + c)(**kwargs)
        self.assertEqual(got, expect)
        [foo_sig] = foo.signatures
        self.assertEqual(foo_sig[2], types.literal(3))

    def test_unsupported_literal_type(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(a, b, c):
            if False:
                i = 10
                return i + 15
            return inner(a, c)

        @njit
        def inner(x, y):
            if False:
                while True:
                    i = 10
            return x + literally(y)
        arr = np.arange(10)
        with self.assertRaises(errors.LiteralTypingError) as raises:
            foo(a=1, b=2, c=arr)
        self.assertIn('numpy.ndarray', str(raises.exception))

    def test_biliteral(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(a, b, c):
            if False:
                print('Hello World!')
            return inner(a, b) + inner(b, c)

        @njit
        def inner(x, y):
            if False:
                return 10
            return x + literally(y)
        kwargs = dict(a=1, b=2, c=3)
        got = foo(**kwargs)
        expect = (lambda a, b, c: a + b + b + c)(**kwargs)
        self.assertEqual(got, expect)
        [(type_a, type_b, type_c)] = foo.signatures
        self.assertNotIsInstance(type_a, types.Literal)
        self.assertIsInstance(type_b, types.Literal)
        self.assertEqual(type_b.literal_value, 2)
        self.assertIsInstance(type_c, types.Literal)
        self.assertEqual(type_c.literal_value, 3)

    def test_literally_varargs(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(a, *args):
            if False:
                for i in range(10):
                    print('nop')
            return literally(args)
        with self.assertRaises(errors.LiteralTypingError):
            foo(1, 2, 3)

        @njit
        def bar(a, b):
            if False:
                for i in range(10):
                    print('nop')
            foo(a, b)
        with self.assertRaises(errors.TypingError) as raises:
            bar(1, 2)
        self.assertIn('Cannot request literal type', str(raises.exception))

    @unittest.expectedFailure
    def test_literally_defaults(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(a, b=1):
            if False:
                print('Hello World!')
            return (a, literally(b))
        foo(a=1)

    @unittest.expectedFailure
    def test_literally_defaults_inner(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(a, b=1):
            if False:
                return 10
            return (a, literally(b))

        @njit
        def bar(a):
            if False:
                for i in range(10):
                    print('nop')
            return foo(a) + 1
        bar(1)

    def test_literally_from_module(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(x):
            if False:
                print('Hello World!')
            return numba.literally(x)
        got = foo(123)
        self.assertEqual(got, foo.py_func(123))
        self.assertIsInstance(foo.signatures[0][0], types.Literal)

    def test_non_literal(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(a, b):
            if False:
                print('Hello World!')
            return literally(1 + a)
        with self.assertRaises(errors.TypingError) as raises:
            foo(1, 2)
        self.assertIn('Invalid use of non-Literal type', str(raises.exception))

    def test_inlined_literal(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(a, b):
            if False:
                while True:
                    i = 10
            v = 1000
            return a + literally(v) + literally(b)
        got = foo(1, 2)
        self.assertEqual(got, foo.py_func(1, 2))

        @njit
        def bar():
            if False:
                while True:
                    i = 10
            a = 100
            b = 9
            return foo(a=b, b=a)
        got = bar()
        self.assertEqual(got, bar.py_func())

    def test_aliased_variable(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(a, b, c):
            if False:
                while True:
                    i = 10

            def closure(d):
                if False:
                    print('Hello World!')
                return literally(d) + 10 * inner(a, b)
            return closure(c)

        @njit
        def inner(x, y):
            if False:
                return 10
            return x + literally(y)
        kwargs = dict(a=1, b=2, c=3)
        got = foo(**kwargs)
        expect = (lambda a, b, c: c + 10 * (a + b))(**kwargs)
        self.assertEqual(got, expect)
        [(type_a, type_b, type_c)] = foo.signatures
        self.assertNotIsInstance(type_a, types.Literal)
        self.assertIsInstance(type_b, types.Literal)
        self.assertEqual(type_b.literal_value, 2)
        self.assertIsInstance(type_c, types.Literal)
        self.assertEqual(type_c.literal_value, 3)

    def test_overload_explicit(self):
        if False:
            for i in range(10):
                print('nop')

        def do_this(x, y):
            if False:
                print('Hello World!')
            return x + y

        @overload(do_this)
        def ov_do_this(x, y):
            if False:
                return 10
            SentryLiteralArgs(['x']).for_function(ov_do_this).bind(x, y)
            return lambda x, y: x + y

        @njit
        def foo(a, b):
            if False:
                while True:
                    i = 10
            return do_this(a, b)
        a = 123
        b = 321
        r = foo(a, b)
        self.assertEqual(r, a + b)
        [type_a, type_b] = foo.signatures[0]
        self.assertIsInstance(type_a, types.Literal)
        self.assertEqual(type_a.literal_value, a)
        self.assertNotIsInstance(type_b, types.Literal)

    def test_overload_implicit(self):
        if False:
            while True:
                i = 10

        def do_this(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        @njit
        def hidden(x, y):
            if False:
                i = 10
                return i + 15
            return literally(x) + y

        @overload(do_this)
        def ov_do_this(x, y):
            if False:
                return 10
            if isinstance(x, types.Integer):
                return lambda x, y: hidden(x, y)

        @njit
        def foo(a, b):
            if False:
                i = 10
                return i + 15
            return do_this(a, b)
        a = 123
        b = 321
        r = foo(a, b)
        self.assertEqual(r, a + b)
        [type_a, type_b] = foo.signatures[0]
        self.assertIsInstance(type_a, types.Literal)
        self.assertEqual(type_a.literal_value, a)
        self.assertNotIsInstance(type_b, types.Literal)

    def test_overload_error_loop(self):
        if False:
            while True:
                i = 10

        def do_this(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        @njit
        def hidden(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return literally(x) + y

        @overload(do_this)
        def ov_do_this(x, y):
            if False:
                i = 10
                return i + 15
            if isinstance(y, types.IntegerLiteral):
                raise errors.NumbaValueError('oops')
            else:

                def impl(x, y):
                    if False:
                        print('Hello World!')
                    return hidden(x, y)
                return impl

        @njit
        def foo(a, b):
            if False:
                while True:
                    i = 10
            return do_this(a, literally(b))
        with self.assertRaises(errors.CompilerError) as raises:
            foo(a=123, b=321)
        self.assertIn('Repeated literal typing request', str(raises.exception))

class TestLiteralDispatchWithCustomType(TestCase):

    def make_dummy_type(self):
        if False:
            i = 10
            return i + 15

        class Dummy(object):

            def lit(self, a):
                if False:
                    while True:
                        i = 10
                return a

        class DummyType(types.Type):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(DummyType, self).__init__(name='dummy')

        @register_model(DummyType)
        class DummyTypeModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                if False:
                    return 10
                members = []
                super(DummyTypeModel, self).__init__(dmm, fe_type, members)

        @intrinsic
        def init_dummy(typingctx):
            if False:
                return 10

            def codegen(context, builder, signature, args):
                if False:
                    for i in range(10):
                        print('nop')
                dummy = cgutils.create_struct_proxy(signature.return_type)(context, builder)
                return dummy._getvalue()
            sig = signature(DummyType())
            return (sig, codegen)

        @overload(Dummy)
        def dummy_overload():
            if False:
                return 10

            def ctor():
                if False:
                    print('Hello World!')
                return init_dummy()
            return ctor
        return (DummyType, Dummy)

    def test_overload_method(self):
        if False:
            return 10
        (DummyType, Dummy) = self.make_dummy_type()

        @overload_method(DummyType, 'lit')
        def lit_overload(self, a):
            if False:
                for i in range(10):
                    print('nop')

            def impl(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                return literally(a)
            return impl

        @njit
        def test_impl(a):
            if False:
                print('Hello World!')
            d = Dummy()
            return d.lit(a)
        self.assertEqual(test_impl(5), 5)

        @njit
        def inside(a):
            if False:
                for i in range(10):
                    print('nop')
            return test_impl(a + 1)
        with self.assertRaises(errors.TypingError) as raises:
            inside(4)
        self.assertIn('Cannot request literal type.', str(raises.exception))
if __name__ == '__main__':
    unittest.main()