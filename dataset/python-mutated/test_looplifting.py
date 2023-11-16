from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
looplift_flags = Flags()
looplift_flags.enable_pyobject = True
looplift_flags.enable_looplift = True
pyobject_looplift_flags = looplift_flags.copy()
pyobject_looplift_flags.enable_pyobject_looplift = True

def compile_isolated(pyfunc, argtypes, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    from numba.core.registry import cpu_target
    kwargs.setdefault('return_type', None)
    kwargs.setdefault('locals', {})
    return compile_extra(cpu_target.typing_context, cpu_target.target_context, pyfunc, argtypes, **kwargs)

def lift1(x):
    if False:
        print('Hello World!')
    a = np.empty(3)
    for i in range(a.size):
        a[i] = x
    return a

def lift2(x):
    if False:
        i = 10
        return i + 15
    a = np.empty((3, 4))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = x
    return a

def lift3(x):
    if False:
        while True:
            i = 10
    _ = object()
    a = np.arange(5, dtype=np.int64)
    c = 0
    for i in range(a.shape[0]):
        c += a[i] * x
    return c

def lift4(x):
    if False:
        for i in range(10):
            print('nop')
    _ = object()
    a = np.arange(5, dtype=np.int64)
    c = 0
    d = 0
    for i in range(a.shape[0]):
        c += a[i] * x
        d += c
    return c + d

def lift5(x):
    if False:
        for i in range(10):
            print('nop')
    _ = object()
    a = np.arange(4)
    for i in range(a.shape[0]):
        if i > 2:
            break
    return a

def lift_gen1(x):
    if False:
        for i in range(10):
            print('nop')
    a = np.empty(3)
    yield 0
    for i in range(a.size):
        a[i] = x
    yield np.sum(a)

def lift_issue2561():
    if False:
        while True:
            i = 10
    np.empty(1)
    for i in range(10):
        for j in range(10):
            return 1
    return 2

def reject1(x):
    if False:
        for i in range(10):
            print('nop')
    a = np.arange(4)
    for i in range(a.shape[0]):
        return a
    return a

def reject_gen1(x):
    if False:
        for i in range(10):
            print('nop')
    _ = object()
    a = np.arange(4)
    for i in range(a.shape[0]):
        yield a[i]

def reject_gen2(x):
    if False:
        return 10
    _ = object()
    a = np.arange(3)
    for i in range(a.size):
        res = a[i] + x
        for j in range(i):
            res = res ** 2
        yield res

def reject_npm1(x):
    if False:
        return 10
    a = np.empty(3, dtype=np.int32)
    for i in range(a.size):
        _ = object()
        a[i] = np.arange(i + 1)[i]
    return a

class TestLoopLifting(MemoryLeakMixin, TestCase):

    def try_lift(self, pyfunc, argtypes):
        if False:
            for i in range(10):
                print('nop')
        from numba.core.registry import cpu_target
        cres = compile_extra(cpu_target.typing_context, cpu_target.target_context, pyfunc, argtypes, return_type=None, flags=looplift_flags, locals={})
        self.assertEqual(len(cres.lifted), 1)
        return cres

    def assert_lifted_native(self, cres):
        if False:
            print('Hello World!')
        jitloop = cres.lifted[0]
        [loopcres] = jitloop.overloads.values()
        self.assertTrue(loopcres.fndesc.native)

    def check_lift_ok(self, pyfunc, argtypes, args):
        if False:
            print('Hello World!')
        '\n        Check that pyfunc can loop-lift even in nopython mode.\n        '
        cres = self.try_lift(pyfunc, argtypes)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assert_lifted_native(cres)
        self.assertPreciseEqual(expected, got)

    def check_lift_generator_ok(self, pyfunc, argtypes, args):
        if False:
            while True:
                i = 10
        '\n        Check that pyfunc (a generator function) can loop-lift even in\n        nopython mode.\n        '
        cres = self.try_lift(pyfunc, argtypes)
        expected = list(pyfunc(*args))
        got = list(cres.entry_point(*args))
        self.assert_lifted_native(cres)
        self.assertPreciseEqual(expected, got)

    def check_no_lift(self, pyfunc, argtypes, args):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check that pyfunc can't loop-lift.\n        "
        cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
        self.assertFalse(cres.lifted)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertPreciseEqual(expected, got)

    def check_no_lift_generator(self, pyfunc, argtypes, args):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check that pyfunc (a generator function) can't loop-lift.\n        "
        cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
        self.assertFalse(cres.lifted)
        expected = list(pyfunc(*args))
        got = list(cres.entry_point(*args))
        self.assertPreciseEqual(expected, got)

    def check_no_lift_nopython(self, pyfunc, argtypes, args):
        if False:
            print('Hello World!')
        '\n        Check that pyfunc will fail loop-lifting if pyobject mode\n        is disabled inside the loop, succeed otherwise.\n        '
        cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
        self.assertTrue(cres.lifted)
        with self.assertTypingError():
            cres.entry_point(*args)
        cres = compile_isolated(pyfunc, argtypes, flags=pyobject_looplift_flags)
        self.assertTrue(cres.lifted)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertPreciseEqual(expected, got)

    def test_lift1(self):
        if False:
            while True:
                i = 10
        self.check_lift_ok(lift1, (types.intp,), (123,))

    def test_lift2(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_lift_ok(lift2, (types.intp,), (123,))

    def test_lift3(self):
        if False:
            i = 10
            return i + 15
        self.check_lift_ok(lift3, (types.intp,), (123,))

    def test_lift4(self):
        if False:
            print('Hello World!')
        self.check_lift_ok(lift4, (types.intp,), (123,))

    def test_lift5(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_lift_ok(lift5, (types.intp,), (123,))

    def test_lift_issue2561(self):
        if False:
            print('Hello World!')
        self.check_lift_ok(lift_issue2561, (), ())

    def test_lift_gen1(self):
        if False:
            return 10
        self.check_lift_generator_ok(lift_gen1, (types.intp,), (123,))

    def test_reject1(self):
        if False:
            print('Hello World!')
        self.check_no_lift(reject1, (types.intp,), (123,))

    def test_reject_gen1(self):
        if False:
            print('Hello World!')
        self.check_no_lift_generator(reject_gen1, (types.intp,), (123,))

    def test_reject_gen2(self):
        if False:
            while True:
                i = 10
        self.check_no_lift_generator(reject_gen2, (types.intp,), (123,))

    def test_reject_npm1(self):
        if False:
            i = 10
            return i + 15
        self.check_no_lift_nopython(reject_npm1, (types.intp,), (123,))

class TestLoopLiftingAnnotate(TestCase):

    def test_annotate_1(self):
        if False:
            print('Hello World!')
        '\n        Verify that annotation works as expected with one lifted loop\n        '
        from numba import jit

        def bar():
            if False:
                return 10
            pass

        def foo(x):
            if False:
                print('Hello World!')
            bar()
            for i in range(x.size):
                x[i] += 1
            return x
        cfoo = jit(foo)
        x = np.arange(10)
        xcopy = x.copy()
        r = cfoo(x)
        np.testing.assert_equal(r, xcopy + 1)
        buf = StringIO()
        cfoo.inspect_types(file=buf)
        annotation = buf.getvalue()
        buf.close()
        self.assertIn('The function contains lifted loops', annotation)
        line = foo.__code__.co_firstlineno + 2
        self.assertIn('Loop at line {line}'.format(line=line), annotation)
        self.assertIn('Has 1 overloads', annotation)

    def test_annotate_2(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that annotation works as expected with two lifted loops\n        '
        from numba import jit

        def bar():
            if False:
                i = 10
                return i + 15
            pass

        def foo(x):
            if False:
                return 10
            bar()
            for i in range(x.size):
                x[i] += 1
            for j in range(x.size):
                x[j] *= 2
            return x
        cfoo = jit(foo)
        x = np.arange(10)
        xcopy = x.copy()
        r = cfoo(x)
        np.testing.assert_equal(r, (xcopy + 1) * 2)
        buf = StringIO()
        cfoo.inspect_types(file=buf)
        annotation = buf.getvalue()
        buf.close()
        self.assertIn('The function contains lifted loops', annotation)
        line1 = foo.__code__.co_firstlineno + 3
        line2 = foo.__code__.co_firstlineno + 6
        self.assertIn('Loop at line {line}'.format(line=line1), annotation)
        self.assertIn('Loop at line {line}'.format(line=line2), annotation)

class TestLoopLiftingInAction(MemoryLeakMixin, TestCase):

    def assert_has_lifted(self, jitted, loopcount):
        if False:
            return 10
        lifted = jitted.overloads[jitted.signatures[0]].lifted
        self.assertEqual(len(lifted), loopcount)

    def test_issue_734(self):
        if False:
            return 10
        from numba import jit, void, int32, double

        @jit(void(int32, double[:]), forceobj=True)
        def forloop_with_if(u, a):
            if False:
                i = 10
                return i + 15
            if u == 0:
                for i in range(a.shape[0]):
                    a[i] = a[i] * 2.0
            else:
                for i in range(a.shape[0]):
                    a[i] = a[i] + 1.0
        for u in (0, 1):
            nb_a = np.arange(10, dtype='int32')
            np_a = np.arange(10, dtype='int32')
            forloop_with_if(u, nb_a)
            forloop_with_if.py_func(u, np_a)
            self.assertPreciseEqual(nb_a, np_a)

    def test_issue_812(self):
        if False:
            return 10
        from numba import jit

        @jit('f8[:](f8[:])', forceobj=True)
        def test(x):
            if False:
                print('Hello World!')
            res = np.zeros(len(x))
            ind = 0
            for ii in range(len(x)):
                ind += 1
                res[ind] = x[ind]
                if x[ind] >= 10:
                    break
            for ii in range(ind + 1, len(x)):
                res[ii] = 0
            return res
        x = np.array([1.0, 4, 2, -3, 5, 2, 10, 5, 2, 6])
        np.testing.assert_equal(test.py_func(x), test(x))

    def test_issue_2368(self):
        if False:
            return 10
        from numba import jit

        def lift_issue2368(a, b):
            if False:
                while True:
                    i = 10
            s = 0
            for e in a:
                s += e
            h = b.__hash__()
            return (s, h)
        a = np.ones(10)
        b = object()
        jitted = jit(lift_issue2368)
        expected = lift_issue2368(a, b)
        got = jitted(a, b)
        self.assertEqual(expected[0], got[0])
        self.assertEqual(expected[1], got[1])
        jitloop = jitted.overloads[jitted.signatures[0]].lifted[0]
        [loopcres] = jitloop.overloads.values()
        self.assertTrue(loopcres.fndesc.native)

    def test_no_iteration_w_redef(self):
        if False:
            i = 10
            return i + 15
        from numba import jit

        @jit(forceobj=True)
        def test(n):
            if False:
                print('Hello World!')
            res = 0
            for i in range(n):
                res = i
            return res
        self.assertEqual(test.py_func(-1), test(-1))
        self.assert_has_lifted(test, loopcount=1)
        self.assertEqual(test.py_func(1), test(1))
        self.assert_has_lifted(test, loopcount=1)

    def test_no_iteration(self):
        if False:
            return 10
        from numba import jit

        @jit(forceobj=True)
        def test(n):
            if False:
                while True:
                    i = 10
            res = 0
            for i in range(n):
                res += i
            return res
        self.assertEqual(test.py_func(-1), test(-1))
        self.assert_has_lifted(test, loopcount=1)
        self.assertEqual(test.py_func(1), test(1))
        self.assert_has_lifted(test, loopcount=1)

    def test_define_in_loop_body(self):
        if False:
            while True:
                i = 10
        from numba import jit

        @jit(forceobj=True)
        def test(n):
            if False:
                i = 10
                return i + 15
            for i in range(n):
                res = i
            return res
        self.assertEqual(test.py_func(1), test(1))
        self.assert_has_lifted(test, loopcount=1)

    def test_invalid_argument(self):
        if False:
            return 10
        "Test a problem caused by invalid discovery of loop argument\n        when a variable is used afterwards but not before.\n\n        Before the fix, this will result in::\n\n        numba.ir.NotDefinedError: 'i' is not defined\n        "
        from numba import jit

        @jit(forceobj=True)
        def test(arg):
            if False:
                i = 10
                return i + 15
            if type(arg) == np.ndarray:
                if arg.ndim == 1:
                    result = 0.0
                    j = 0
                    for i in range(arg.shape[0]):
                        pass
                else:
                    raise Exception
            else:
                result = 0.0
                (i, j) = (0, 0)
                return result
        arg = np.arange(10)
        self.assertEqual(test.py_func(arg), test(arg))

    def test_conditionally_defined_in_loop(self):
        if False:
            return 10
        from numba import jit

        @jit(forceobj=True)
        def test():
            if False:
                print('Hello World!')
            x = 5
            y = 0
            for i in range(2):
                if i > 0:
                    x = 6
                y += x
            return (y, x)
        self.assertEqual(test.py_func(), test())
        self.assert_has_lifted(test, loopcount=1)

    def test_stack_offset_error_when_has_no_return(self):
        if False:
            for i in range(10):
                print('nop')
        from numba import jit
        import warnings

        def pyfunc(a):
            if False:
                for i in range(10):
                    print('nop')
            if a:
                for i in range(10):
                    pass
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cfunc = jit(forceobj=True)(pyfunc)
            self.assertEqual(pyfunc(True), cfunc(True))

    def test_variable_scope_bug(self):
        if False:
            print('Hello World!')
        '\n        https://github.com/numba/numba/issues/2179\n\n        Looplifting transformation is using the wrong version of variable `h`.\n        '
        from numba import jit

        def bar(x):
            if False:
                i = 10
                return i + 15
            return x

        def foo(x):
            if False:
                print('Hello World!')
            h = 0.0
            for k in range(x):
                h = h + k
            h = h - bar(x)
            return h
        cfoo = jit(foo)
        self.assertEqual(foo(10), cfoo(10))

    def test_recompilation_loop(self):
        if False:
            print('Hello World!')
        '\n        https://github.com/numba/numba/issues/2481\n        '
        from numba import jit

        def foo(x, y):
            if False:
                for i in range(10):
                    print('nop')
            A = x[::y]
            c = 1
            for k in range(A.size):
                object()
                c = c * A[::-1][k]
            return c
        cfoo = jit(foo)
        args = (np.arange(10), 1)
        self.assertEqual(foo(*args), cfoo(*args))
        self.assertEqual(len(cfoo.overloads[cfoo.signatures[0]].lifted), 1)
        lifted = cfoo.overloads[cfoo.signatures[0]].lifted[0]
        self.assertEqual(len(lifted.signatures), 1)
        args = (np.arange(10), -1)
        self.assertEqual(foo(*args), cfoo(*args))
        self.assertEqual(len(lifted.signatures), 2)

    def test_lift_listcomp_block0(self):
        if False:
            i = 10
            return i + 15

        def foo(X):
            if False:
                i = 10
                return i + 15
            [y for y in (1,)]
            for x in (1,):
                pass
            return X
        from numba import jit
        f = jit()(foo)
        f(1)
        self.assertEqual(f.overloads[f.signatures[0]].lifted, ())
        f = jit(forceobj=True)(foo)
        f(1)
        self.assertEqual(len(f.overloads[f.signatures[0]].lifted), 1)

    def test_lift_objectmode_issue_4223(self):
        if False:
            while True:
                i = 10
        from numba import jit

        @jit
        def foo(a, b, c, d, x0, y0, n):
            if False:
                i = 10
                return i + 15
            (xs, ys) = (np.zeros(n), np.zeros(n))
            (xs[0], ys[0]) = (x0, y0)
            for i in np.arange(n - 1):
                xs[i + 1] = np.sin(a * ys[i]) + c * np.cos(a * xs[i])
                ys[i + 1] = np.sin(b * xs[i]) + d * np.cos(b * ys[i])
            object()
            return (xs, ys)
        kwargs = dict(a=1.7, b=1.7, c=0.6, d=1.2, x0=0, y0=0, n=200)
        got = foo(**kwargs)
        expected = foo.py_func(**kwargs)
        self.assertPreciseEqual(got[0], expected[0])
        self.assertPreciseEqual(got[1], expected[1])
        [lifted] = foo.overloads[foo.signatures[0]].lifted
        self.assertEqual(len(lifted.nopython_signatures), 1)
if __name__ == '__main__':
    unittest.main()