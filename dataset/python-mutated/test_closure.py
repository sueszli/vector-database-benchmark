import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline

class TestClosure(TestCase):

    def run_jit_closure_variable(self, **jitargs):
        if False:
            while True:
                i = 10
        Y = 10

        def add_Y(x):
            if False:
                return 10
            return x + Y
        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)
        Y = 12
        self.assertEqual(c_add_Y(1), 11)

    def test_jit_closure_variable(self):
        if False:
            print('Hello World!')
        self.run_jit_closure_variable(forceobj=True)

    def test_jit_closure_variable_npm(self):
        if False:
            print('Hello World!')
        self.run_jit_closure_variable(nopython=True)

    def run_rejitting_closure(self, **jitargs):
        if False:
            return 10
        Y = 10

        def add_Y(x):
            if False:
                print('Hello World!')
            return x + Y
        c_add_Y = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y(1), 11)
        Y = 12
        c_add_Y_2 = jit('i4(i4)', **jitargs)(add_Y)
        self.assertEqual(c_add_Y_2(1), 13)
        Y = 13
        self.assertEqual(c_add_Y_2(1), 13)
        self.assertEqual(c_add_Y(1), 11)

    def test_rejitting_closure(self):
        if False:
            i = 10
            return i + 15
        self.run_rejitting_closure(forceobj=True)

    def test_rejitting_closure_npm(self):
        if False:
            while True:
                i = 10
        self.run_rejitting_closure(nopython=True)

    def run_jit_multiple_closure_variables(self, **jitargs):
        if False:
            i = 10
            return i + 15
        Y = 10
        Z = 2

        def add_Y_mult_Z(x):
            if False:
                print('Hello World!')
            return (x + Y) * Z
        c_add_Y_mult_Z = jit('i4(i4)', **jitargs)(add_Y_mult_Z)
        self.assertEqual(c_add_Y_mult_Z(1), 22)

    def test_jit_multiple_closure_variables(self):
        if False:
            i = 10
            return i + 15
        self.run_jit_multiple_closure_variables(forceobj=True)

    def test_jit_multiple_closure_variables_npm(self):
        if False:
            while True:
                i = 10
        self.run_jit_multiple_closure_variables(nopython=True)

    def run_jit_inner_function(self, **jitargs):
        if False:
            while True:
                i = 10

        def mult_10(a):
            if False:
                i = 10
                return i + 15
            return a * 10
        c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
        c_mult_10.disable_compile()

        def do_math(x):
            if False:
                i = 10
                return i + 15
            return c_mult_10(x + 4)
        c_do_math = jit('intp(intp)', **jitargs)(do_math)
        c_do_math.disable_compile()
        with self.assertRefCount(c_do_math, c_mult_10):
            self.assertEqual(c_do_math(1), 50)

    def test_jit_inner_function(self):
        if False:
            print('Hello World!')
        self.run_jit_inner_function(forceobj=True)

    def test_jit_inner_function_npm(self):
        if False:
            print('Hello World!')
        self.run_jit_inner_function(nopython=True)

class TestInlinedClosure(TestCase):
    """
    Tests for (partial) closure support in njit. The support is partial
    because it only works for closures that can be successfully inlined
    at compile time.
    """

    def test_inner_function(self):
        if False:
            while True:
                i = 10

        def outer(x):
            if False:
                print('Hello World!')

            def inner(x):
                if False:
                    print('Hello World!')
                return x * x
            return inner(x) + inner(x)
        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure(self):
        if False:
            i = 10
            return i + 15

        def outer(x):
            if False:
                i = 10
                return i + 15
            y = x + 1

            def inner(x):
                if False:
                    while True:
                        i = 10
                return x * x + y
            return inner(x) + inner(x)
        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure_2(self):
        if False:
            while True:
                i = 10

        def outer(x):
            if False:
                return 10
            y = x + 1

            def inner(x):
                if False:
                    print('Hello World!')
                return x * y
            y = inner(x)
            return y + inner(x)
        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_inner_function_with_closure_3(self):
        if False:
            print('Hello World!')
        code = '\n            def outer(x):\n                y = x + 1\n                z = 0\n\n                def inner(x):\n                    nonlocal z\n                    z += x * x\n                    return z + y\n\n                return inner(x) + inner(x) + z\n        '
        ns = {}
        exec(code.strip(), ns)
        cfunc = njit(ns['outer'])
        self.assertEqual(cfunc(10), ns['outer'](10))

    def test_inner_function_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def outer(x):
            if False:
                print('Hello World!')

            def inner(y):
                if False:
                    i = 10
                    return i + 15

                def innermost(z):
                    if False:
                        print('Hello World!')
                    return x + y + z
                s = 0
                for i in range(y):
                    s += innermost(i)
                return s
            return inner(x * x)
        cfunc = njit(outer)
        self.assertEqual(cfunc(10), outer(10))

    def test_bulk_use_cases(self):
        if False:
            print('Hello World!')
        ' Tests the large number of use cases defined below '

        @njit
        def fib3(n):
            if False:
                print('Hello World!')
            if n < 2:
                return n
            return fib3(n - 1) + fib3(n - 2)

        def outer1(x):
            if False:
                print('Hello World!')
            ' Test calling recursive function from inner '

            def inner(x):
                if False:
                    for i in range(10):
                        print('nop')
                return fib3(x)
            return inner(x)

        def outer2(x):
            if False:
                i = 10
                return i + 15
            ' Test calling recursive function from closure '
            z = x + 1

            def inner(x):
                if False:
                    print('Hello World!')
                return x + fib3(z)
            return inner(x)

        def outer3(x):
            if False:
                print('Hello World!')
            ' Test recursive inner '

            def inner(x):
                if False:
                    return 10
                if x < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer4(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test recursive closure '
            y = x + 1

            def inner(x):
                if False:
                    return 10
                if x + y < 2:
                    return 10
                else:
                    inner(x - 1)
            return inner(x)

        def outer5(x):
            if False:
                print('Hello World!')
            ' Test nested closure '
            y = x + 1

            def inner1(x):
                if False:
                    return 10
                z = y + x + 2

                def inner2(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x + z
                return inner2(x) + y
            return inner1(x)

        def outer6(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test closure with list comprehension in body '
            y = x + 1

            def inner1(x):
                if False:
                    return 10
                z = y + x + 2
                return [t for t in range(z)]
            return inner1(x)
        _OUTER_SCOPE_VAR = 9

        def outer7(x):
            if False:
                print('Hello World!')
            ' Test use of outer scope var, no closure '
            z = x + 1
            return x + z + _OUTER_SCOPE_VAR
        _OUTER_SCOPE_VAR = 9

        def outer8(x):
            if False:
                i = 10
                return i + 15
            ' Test use of outer scope var, with closure '
            z = x + 1

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + z + _OUTER_SCOPE_VAR
            return inner(x)

        def outer9(x):
            if False:
                i = 10
                return i + 15
            ' Test closure assignment'
            z = x + 1

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + z
            f = inner
            return f(x)

        def outer10(x):
            if False:
                i = 10
                return i + 15
            ' Test two inner, one calls other '
            z = x + 1

            def inner(x):
                if False:
                    while True:
                        i = 10
                return x + z

            def inner2(x):
                if False:
                    return 10
                return inner(x)
            return inner2(x)

        def outer11(x):
            if False:
                return 10
            ' return the closure '
            z = x + 1

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + z
            return inner

        def outer12(x):
            if False:
                i = 10
                return i + 15
            ' closure with kwarg'
            z = x + 1

            def inner(x, kw=7):
                if False:
                    return 10
                return x + z + kw
            return inner(x)

        def outer13(x, kw=7):
            if False:
                i = 10
                return i + 15
            ' outer with kwarg no closure'
            z = x + 1 + kw
            return z

        def outer14(x, kw=7):
            if False:
                for i in range(10):
                    print('nop')
            ' outer with kwarg used in closure'
            z = x + 1

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + z + kw
            return inner(x)

        def outer15(x, kw=7):
            if False:
                while True:
                    i = 10
            ' outer with kwarg as arg to closure'
            z = x + 1

            def inner(x, kw):
                if False:
                    while True:
                        i = 10
                return x + z + kw
            return inner(x, kw)

        def outer16(x):
            if False:
                for i in range(10):
                    print('nop')
            ' closure is generator, consumed locally '
            z = x + 1

            def inner(x):
                if False:
                    return 10
                yield (x + z)
            return list(inner(x))

        def outer17(x):
            if False:
                i = 10
                return i + 15
            ' closure is generator, returned '
            z = x + 1

            def inner(x):
                if False:
                    for i in range(10):
                        print('nop')
                yield (x + z)
            return inner(x)

        def outer18(x):
            if False:
                return 10
            ' closure is generator, consumed in loop '
            z = x + 1

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                yield (x + z)
            for i in inner(x):
                t = i
            return t

        def outer19(x):
            if False:
                i = 10
                return i + 15
            ' closure as arg to another closure '
            z1 = x + 1
            z2 = x + 2

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + z1

            def inner2(f, x):
                if False:
                    return 10
                return f(x) + z2
            return inner2(inner, x)

        def outer20(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test calling numpy in closure '
            z = x + 1

            def inner(x):
                if False:
                    return 10
                return x + numpy.cos(z)
            return inner(x)

        def outer21(x):
            if False:
                while True:
                    i = 10
            ' Test calling numpy import as in closure '
            z = x + 1

            def inner(x):
                if False:
                    while True:
                        i = 10
                return x + np.cos(z)
            return inner(x)

        def outer22():
            if False:
                return 10
            'Test to ensure that unsupported *args raises correctly'

            def bar(a, b):
                if False:
                    i = 10
                    return i + 15
                pass
            x = (1, 2)
            bar(*x)
        f = [outer1, outer2, outer5, outer6, outer7, outer8, outer9, outer10, outer12, outer13, outer14, outer15, outer19, outer20, outer21]
        for ref in f:
            cfunc = njit(ref)
            var = 10
            self.assertEqual(cfunc(var), ref(var))
        with self.assertRaises(NotImplementedError) as raises:
            cfunc = jit(nopython=True)(outer3)
            cfunc(var)
        msg = 'Unsupported use of op_LOAD_CLOSURE encountered'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(NotImplementedError) as raises:
            cfunc = jit(nopython=True)(outer4)
            cfunc(var)
        msg = 'Unsupported use of op_LOAD_CLOSURE encountered'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(outer11)
            cfunc(var)
        msg = 'Cannot capture the non-constant value'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer16)
            cfunc(var)
        msg = 'The use of yield in a closure is unsupported.'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer17)
            cfunc(var)
        msg = 'The use of yield in a closure is unsupported.'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer18)
            cfunc(var)
        msg = 'The use of yield in a closure is unsupported.'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(UnsupportedError) as raises:
            cfunc = jit(nopython=True)(outer22)
            cfunc()
        msg = 'Calling a closure with *args is unsupported.'
        self.assertIn(msg, str(raises.exception))

    def test_closure_renaming_scheme(self):
        if False:
            while True:
                i = 10

        @njit(pipeline_class=IRPreservingTestPipeline)
        def foo(a, b):
            if False:
                i = 10
                return i + 15

            def bar(z):
                if False:
                    return 10
                x = 5
                y = 10
                return x + y + z
            return (bar(a), bar(b))
        self.assertEqual(foo(10, 20), (25, 35))
        func_ir = foo.overloads[foo.signatures[0]].metadata['preserved_ir']
        store = []
        for blk in func_ir.blocks.values():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Const):
                        if stmt.value.value == 5:
                            store.append(stmt)
        self.assertEqual(len(store), 2)
        for i in store:
            name = i.target.name
            regex = 'closure__locals__bar_v[0-9]+.x'
            self.assertRegex(name, regex)

    def test_issue9222(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                print('Hello World!')

            def bar(x, y=1.1):
                if False:
                    while True:
                        i = 10
                return x + y
            return bar

        @njit
        def consume():
            if False:
                while True:
                    i = 10
            return foo()(4)
        np.testing.assert_allclose(consume(), 4 + 1.1)

class TestObjmodeFallback(TestCase):
    decorators = [jit, jit(forceobj=True)]

    def test_issue2955(self):
        if False:
            return 10

        def numbaFailure(scores, cooc):
            if False:
                print('Hello World!')
            (rows, cols) = scores.shape
            for i in range(rows):
                coxv = scores[i]
                groups = sorted(set(coxv), reverse=True)
                [set(np.argwhere(coxv == x).flatten()) for x in groups]
        x = np.random.random((10, 10))
        y = np.abs(np.random.randn(10, 10) * 1.732).astype(int)
        for d in self.decorators:
            d(numbaFailure)(x, y)

    def test_issue3239(self):
        if False:
            i = 10
            return i + 15

        def fit(X, y):
            if False:
                print('Hello World!')
            if type(X) is not np.ndarray:
                X = np.array(X)
            if type(y) is not np.ndarray:
                y = np.array(y)
            (m, _) = X.shape
            X = np.hstack((np.array([[1] for _ in range(m)]), X))
            res = np.dot(np.dot(X, X.T), y)
            intercept = res[0]
            coefs = res[1:]
            return (intercept, coefs)
        for d in self.decorators:
            res = d(fit)(np.arange(10).reshape(1, 10), np.arange(10).reshape(1, 10))
            exp = fit(np.arange(10).reshape(1, 10), np.arange(10).reshape(1, 10))
            np.testing.assert_equal(res, exp)

    def test_issue3289(self):
        if False:
            return 10
        b = [(5, 124), (52, 5)]

        def a():
            if False:
                for i in range(10):
                    print('nop')
            [b[index] for index in [0, 1]]
            for x in range(5):
                pass
        for d in self.decorators:
            d(a)()

    def test_issue3413(self):
        if False:
            print('Hello World!')

        def foo(data):
            if False:
                for i in range(10):
                    print('nop')
            t = max([len(m) for m in data['y']])
            mask = data['x'] == 0
            if any(mask):
                z = 15
            return (t, z)
        data = {'x': np.arange(5), 'y': [[1], [2, 3]]}
        for d in self.decorators:
            res = d(foo)(data)
            np.testing.assert_allclose(res, foo(data))

    def test_issue3659(self):
        if False:
            return 10

        def main():
            if False:
                print('Hello World!')
            a = np.array(((1, 2), (3, 4)))
            return np.array([x for x in a])
        for d in self.decorators:
            res = d(main)()
            np.testing.assert_allclose(res, main())

    def test_issue3803(self):
        if False:
            print('Hello World!')

        def center(X):
            if False:
                return 10
            np.array([np.float_(x) for x in X.T])
            np.array([np.float_(1) for _ in X.T])
            return X
        X = np.zeros((10,))
        for d in self.decorators:
            res = d(center)(X)
            np.testing.assert_allclose(res, center(X))
if __name__ == '__main__':
    unittest.main()