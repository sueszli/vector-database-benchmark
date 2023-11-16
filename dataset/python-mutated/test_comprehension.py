import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba.core.compiler import compile_isolated
from numba import jit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
PARALLEL_SUPPORTED = not _32bit

def comp_list(n):
    if False:
        for i in range(10):
            print('nop')
    l = [i for i in range(n)]
    s = 0
    for i in l:
        s += i
    return s

class TestListComprehension(TestCase):

    def test_comp_list(self):
        if False:
            i = 10
            return i + 15
        pyfunc = comp_list
        cres = compile_isolated(pyfunc, [types.intp])
        cfunc = cres.entry_point
        self.assertEqual(cfunc(5), pyfunc(5))
        self.assertEqual(cfunc(0), pyfunc(0))
        self.assertEqual(cfunc(-1), pyfunc(-1))

    def test_bulk_use_cases(self):
        if False:
            i = 10
            return i + 15
        ' Tests the large number of use cases defined below '

        @jit(nopython=True)
        def fib3(n):
            if False:
                i = 10
                return i + 15
            if n < 2:
                return n
            return fib3(n - 1) + fib3(n - 2)

        def list1(x):
            if False:
                return 10
            ' Test basic list comprehension '
            return [i for i in range(1, len(x) - 1)]

        def list2(x):
            if False:
                while True:
                    i = 10
            ' Test conditional list comprehension '
            return [y for y in x if y < 2]

        def list3(x):
            if False:
                while True:
                    i = 10
            ' Test ternary list comprehension '
            return [y if y < 2 else -1 for y in x]

        def list4(x):
            if False:
                print('Hello World!')
            ' Test list comprehension to np.array ctor '
            return np.array([1, 2, 3])

        def list5(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test nested list comprehension to np.array ctor '
            return np.array([np.array([z for z in x]) for y in x])

        def list6(x):
            if False:
                return 10
            ' Test use of inner function in list comprehension '

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return x + 1
            return [inner(z) for z in x]

        def list7(x):
            if False:
                print('Hello World!')
            ' Test use of closure in list comprehension '
            y = 3

            def inner(x):
                if False:
                    return 10
                return x + y
            return [inner(z) for z in x]

        def list8(x):
            if False:
                i = 10
                return i + 15
            ' Test use of list comprehension as arg to inner function '
            l = [z + 1 for z in x]

            def inner(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x[0] + 1
            q = inner(l)
            return q

        def list9(x):
            if False:
                print('Hello World!')
            ' Test use of list comprehension access in closure '
            l = [z + 1 for z in x]

            def inner(x):
                if False:
                    while True:
                        i = 10
                return x[0] + l[1]
            return inner(x)

        def list10(x):
            if False:
                return 10
            ' Test use of list comprehension access in closure and as arg '
            l = [z + 1 for z in x]

            def inner(x):
                if False:
                    i = 10
                    return i + 15
                return [y + l[0] for y in x]
            return inner(l)

        def list11(x):
            if False:
                while True:
                    i = 10
            ' Test scalar array construction in list comprehension '
            l = [np.array(z) for z in x]
            return l

        def list12(x):
            if False:
                i = 10
                return i + 15
            ' Test scalar type conversion construction in list comprehension '
            l = [np.float64(z) for z in x]
            return l

        def list13(x):
            if False:
                return 10
            ' Test use of explicit numpy scalar ctor reference in list comprehension '
            l = [numpy.float64(z) for z in x]
            return l

        def list14(x):
            if False:
                i = 10
                return i + 15
            ' Test use of python scalar ctor reference in list comprehension '
            l = [float(z) for z in x]
            return l

        def list15(x):
            if False:
                return 10
            ' Test use of python scalar ctor reference in list comprehension followed by np array construction from the list'
            l = [float(z) for z in x]
            return np.array(l)

        def list16(x):
            if False:
                while True:
                    i = 10
            ' Test type unification from np array ctors consuming list comprehension '
            l1 = [float(z) for z in x]
            l2 = [z for z in x]
            ze = np.array(l1)
            oe = np.array(l2)
            return ze + oe

        def list17(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test complex list comprehension including math calls '
            return [(a, b, c) for a in x for b in x for c in x if np.sqrt(a ** 2 + b ** 2) == c]
        _OUTER_SCOPE_VAR = 9

        def list18(x):
            if False:
                return 10
            ' Test loop list with outer scope var as conditional'
            z = []
            for i in x:
                if i < _OUTER_SCOPE_VAR:
                    z.append(i)
            return z
        _OUTER_SCOPE_VAR = 9

        def list19(x):
            if False:
                print('Hello World!')
            ' Test list comprehension with outer scope as conditional'
            return [i for i in x if i < _OUTER_SCOPE_VAR]

        def list20(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test return empty list '
            return [i for i in x if i == -1000]

        def list21(x):
            if False:
                print('Hello World!')
            ' Test call a jitted function in a list comprehension '
            return [fib3(i) for i in x]

        def list22(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test create two lists comprehensions and a third walking the first two '
            a = [y - 1 for y in x]
            b = [y + 1 for y in x]
            return [x for x in a for y in b if x == y]

        def list23(x):
            if False:
                for i in range(10):
                    print('nop')
            ' Test operation on comprehension generated list '
            z = [y for y in x]
            z.append(1)
            return z

        def list24(x):
            if False:
                while True:
                    i = 10
            ' Test type promotion '
            z = [float(y) if y > 3 else y for y in x]
            return z

        def list25(x):
            if False:
                for i in range(10):
                    print('nop')
            included = np.array([1, 2, 6, 8])
            not_included = [i for i in range(10) if i not in list(included)]
            return not_included
        f = [list1, list2, list3, list4, list6, list7, list8, list9, list10, list11, list12, list13, list14, list15, list16, list17, list18, list19, list20, list21, list22, list23, list24, list25]
        var = [1, 2, 3, 4, 5]
        for ref in f:
            try:
                cfunc = jit(nopython=True)(ref)
                self.assertEqual(cfunc(var), ref(var))
            except ValueError:
                try:
                    np.testing.assert_allclose(cfunc(var), ref(var))
                except Exception:
                    raise
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(list5)
            cfunc(var)
        if sys.maxsize > 2 ** 32:
            bits = 64
        else:
            bits = 32

    def test_objmode_inlining(self):
        if False:
            print('Hello World!')

        def objmode_func(y):
            if False:
                return 10
            z = object()
            inlined = [x for x in y]
            return inlined
        cfunc = jit(forceobj=True)(objmode_func)
        t = [1, 2, 3]
        expected = objmode_func(t)
        got = cfunc(t)
        self.assertPreciseEqual(expected, got)

class TestArrayComprehension(unittest.TestCase):
    _numba_parallel_test_ = False

    def check(self, pyfunc, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'A generic check function that run both pyfunc, and jitted pyfunc,\n        and compare results.'
        run_parallel = kwargs.get('run_parallel', False)
        assert_allocate_list = kwargs.get('assert_allocate_list', False)
        assert_dtype = kwargs.get('assert_dtype', False)
        cfunc = jit(nopython=True, parallel=run_parallel)(pyfunc)
        pyres = pyfunc(*args)
        cres = cfunc(*args)
        np.testing.assert_array_equal(pyres, cres)
        if assert_dtype:
            self.assertEqual(cres[1].dtype, assert_dtype)
        if assert_allocate_list:
            self.assertIn('allocate list', cfunc.inspect_llvm(cfunc.signatures[0]))
        else:
            self.assertNotIn('allocate list', cfunc.inspect_llvm(cfunc.signatures[0]))
        if run_parallel:
            self.assertIn('@do_scheduling', cfunc.inspect_llvm(cfunc.signatures[0]))

    def test_comp_with_array_1(self):
        if False:
            print('Hello World!')

        def comp_with_array_1(n):
            if False:
                return 10
            m = n * 2
            l = np.array([i + m for i in range(n)])
            return l
        self.check(comp_with_array_1, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_with_array_1, 5, run_parallel=True)

    def test_comp_with_array_2(self):
        if False:
            print('Hello World!')

        def comp_with_array_2(n, threshold):
            if False:
                for i in range(10):
                    print('nop')
            A = np.arange(-n, n)
            return np.array([x * x if x < threshold else x * 2 for x in A])
        self.check(comp_with_array_2, 5, 0)

    def test_comp_with_array_noinline(self):
        if False:
            i = 10
            return i + 15

        def comp_with_array_noinline(n):
            if False:
                for i in range(10):
                    print('nop')
            m = n * 2
            l = np.array([i + m for i in range(n)])
            return l
        import numba.core.inline_closurecall as ic
        try:
            ic.enable_inline_arraycall = False
            self.check(comp_with_array_noinline, 5, assert_allocate_list=True)
        finally:
            ic.enable_inline_arraycall = True

    def test_comp_with_array_noinline_issue_6053(self):
        if False:
            i = 10
            return i + 15

        def comp_with_array_noinline(n):
            if False:
                for i in range(10):
                    print('nop')
            lst = [0]
            for i in range(n):
                lst.append(i)
            l = np.array(lst)
            return l
        self.check(comp_with_array_noinline, 5, assert_allocate_list=True)

    def test_comp_nest_with_array(self):
        if False:
            for i in range(10):
                print('nop')

        def comp_nest_with_array(n):
            if False:
                i = 10
                return i + 15
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(comp_nest_with_array, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_nest_with_array, 5, run_parallel=True)

    def test_comp_nest_with_array_3(self):
        if False:
            print('Hello World!')

        def comp_nest_with_array_3(n):
            if False:
                print('Hello World!')
            l = np.array([[[i * j * k for k in range(n)] for j in range(n)] for i in range(n)])
            return l
        self.check(comp_nest_with_array_3, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_nest_with_array_3, 5, run_parallel=True)

    def test_comp_nest_with_array_noinline(self):
        if False:
            i = 10
            return i + 15

        def comp_nest_with_array_noinline(n):
            if False:
                while True:
                    i = 10
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        import numba.core.inline_closurecall as ic
        try:
            ic.enable_inline_arraycall = False
            self.check(comp_nest_with_array_noinline, 5, assert_allocate_list=True)
        finally:
            ic.enable_inline_arraycall = True

    def test_comp_with_array_range(self):
        if False:
            return 10

        def comp_with_array_range(m, n):
            if False:
                while True:
                    i = 10
            l = np.array([i for i in range(m, n)])
            return l
        self.check(comp_with_array_range, 5, 10)

    def test_comp_with_array_range_and_step(self):
        if False:
            while True:
                i = 10

        def comp_with_array_range_and_step(m, n):
            if False:
                print('Hello World!')
            l = np.array([i for i in range(m, n, 2)])
            return l
        self.check(comp_with_array_range_and_step, 5, 10)

    def test_comp_with_array_conditional(self):
        if False:
            for i in range(10):
                print('nop')

        def comp_with_array_conditional(n):
            if False:
                return 10
            l = np.array([i for i in range(n) if i % 2 == 1])
            return l
        self.check(comp_with_array_conditional, 10, assert_allocate_list=True)

    def test_comp_nest_with_array_conditional(self):
        if False:
            print('Hello World!')

        def comp_nest_with_array_conditional(n):
            if False:
                print('Hello World!')
            l = np.array([[i * j for j in range(n)] for i in range(n) if i % 2 == 1])
            return l
        self.check(comp_nest_with_array_conditional, 5, assert_allocate_list=True)

    @unittest.skipUnless(numpy_version < (1, 24), 'Setting an array element with a sequence is removed in NumPy 1.24')
    def test_comp_nest_with_dependency(self):
        if False:
            for i in range(10):
                print('nop')

        def comp_nest_with_dependency(n):
            if False:
                return 10
            l = np.array([[i * j for j in range(i + 1)] for i in range(n)])
            return l
        with self.assertRaises(TypingError) as raises:
            self.check(comp_nest_with_dependency, 5)
        self.assertIn(_header_lead, str(raises.exception))
        self.assertIn('array(undefined,', str(raises.exception))

    def test_comp_unsupported_iter(self):
        if False:
            return 10

        def comp_unsupported_iter():
            if False:
                print('Hello World!')
            val = zip([1, 2, 3], [4, 5, 6])
            return np.array([a for (a, b) in val])
        with self.assertRaises(TypingError) as raises:
            self.check(comp_unsupported_iter)
        self.assertIn(_header_lead, str(raises.exception))
        self.assertIn('Unsupported iterator found in array comprehension', str(raises.exception))

    def test_no_array_comp(self):
        if False:
            return 10

        def no_array_comp1(n):
            if False:
                i = 10
                return i + 15
            l = [1, 2, 3, 4]
            a = np.array(l)
            return a
        self.check(no_array_comp1, 10, assert_allocate_list=False)

        def no_array_comp2(n):
            if False:
                print('Hello World!')
            l = [1, 2, 3, 4]
            a = np.array(l)
            l.append(5)
            return a
        self.check(no_array_comp2, 10, assert_allocate_list=True)

    def test_nested_array(self):
        if False:
            print('Hello World!')

        def nested_array(n):
            if False:
                for i in range(10):
                    print('nop')
            l = np.array([np.array([x for x in range(n)]) for y in range(n)])
            return l
        self.check(nested_array, 10)

    def test_nested_array_with_const(self):
        if False:
            for i in range(10):
                print('nop')

        def nested_array(n):
            if False:
                for i in range(10):
                    print('nop')
            l = np.array([np.array([x for x in range(3)]) for y in range(4)])
            return l
        self.check(nested_array, 0)

    def test_array_comp_with_iter(self):
        if False:
            print('Hello World!')

        def array_comp(a):
            if False:
                i = 10
                return i + 15
            l = np.array([x * x for x in a])
            return l
        l = [1, 2, 3, 4, 5]
        self.check(array_comp, l)
        self.check(array_comp, np.array(l))
        self.check(array_comp, tuple(l))
        self.check(array_comp, typed.List(l))

    def test_array_comp_with_dtype(self):
        if False:
            print('Hello World!')

        def array_comp(n):
            if False:
                return 10
            l = np.array([i for i in range(n)], dtype=np.complex64)
            return l
        self.check(array_comp, 10, assert_dtype=np.complex64)

    def test_array_comp_inferred_dtype(self):
        if False:
            print('Hello World!')

        def array_comp(n):
            if False:
                i = 10
                return i + 15
            l = np.array([i * 1j for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_nested(self):
        if False:
            return 10

        def array_comp(n):
            if False:
                while True:
                    i = 10
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_nested_sum(self):
        if False:
            while True:
                i = 10

        def array_comp(n):
            if False:
                i = 10
                return i + 15
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_outside_setitem(self):
        if False:
            for i in range(10):
                print('nop')

        def array_comp(n, v):
            if False:
                print('Hello World!')
            arr = np.array([i for i in range(n)])
            arr[0] = v
            return arr
        v = 1.2
        self.check(array_comp, 10, v, assert_dtype=np.intp)
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(array_comp)
            cfunc(10, 2.3j)
        self.assertIn(_header_lead + ' Function({})'.format(operator.setitem), str(raises.exception))
        self.assertIn('(array({}, 1d, C), Literal[int](0), complex128)'.format(types.intp), str(raises.exception))

    def test_array_comp_shuffle_sideeffect(self):
        if False:
            for i in range(10):
                print('nop')
        nelem = 100

        @jit(nopython=True)
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            numbers = np.array([i for i in range(nelem)])
            np.random.shuffle(numbers)
            print(numbers)
        with captured_stdout() as gotbuf:
            foo()
        got = gotbuf.getvalue().strip()
        with captured_stdout() as expectbuf:
            print(np.array([i for i in range(nelem)]))
        expect = expectbuf.getvalue().strip()
        self.assertNotEqual(got, expect)
        self.assertRegexpMatches(got, '\\[(\\s*\\d+)+\\]')

    def test_empty_list_not_removed(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            t = []
            myList = np.array([1])
            a = np.random.choice(myList, 1)
            t.append(x + a)
            return a
        self.check(f, 5, assert_allocate_list=True)

    def test_reuse_of_array_var(self):
        if False:
            i = 10
            return i + 15
        ' Test issue 3742 '

        def foo(n):
            if False:
                for i in range(10):
                    print('nop')
            [i for i in range(1)]
            z = np.empty(n)
            for i in range(n):
                z = np.zeros(n)
                z[i] = i
            return z
        self.check(foo, 10, assert_allocate_list=True)
if __name__ == '__main__':
    unittest.main()