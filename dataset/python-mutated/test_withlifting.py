import copy
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import typing, errors, cpu
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import MemoryLeak, TestCase, captured_stdout, skip_unless_scipy, linux_only, strace_supported, strace, expected_failure_py311
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest

def get_func_ir(func):
    if False:
        while True:
            i = 10
    func_id = FunctionIdentity.from_function(func)
    bc = ByteCode(func_id=func_id)
    interp = Interpreter(func_id)
    func_ir = interp.interpret(bc)
    return func_ir

def lift1():
    if False:
        for i in range(10):
            print('nop')
    print('A')
    with bypass_context:
        print('B')
        b()
    print('C')

def lift2():
    if False:
        return 10
    x = 1
    print('A', x)
    x = 1
    with bypass_context:
        print('B', x)
        x += 100
        b()
    x += 1
    with bypass_context:
        print('C', x)
        b()
        x += 10
    x += 1
    print('D', x)

def lift3():
    if False:
        print('Hello World!')
    x = 1
    y = 100
    print('A', x, y)
    with bypass_context:
        print('B')
        b()
        x += 100
        with bypass_context:
            print('C')
            y += 100000
            b()
    x += 1
    y += 1
    print('D', x, y)

def lift4():
    if False:
        for i in range(10):
            print('nop')
    x = 0
    print('A', x)
    x += 10
    with bypass_context:
        print('B')
        b()
        x += 1
        for i in range(10):
            with bypass_context:
                print('C')
                b()
                x += i
    with bypass_context:
        print('D')
        b()
        if x:
            x *= 10
    x += 1
    print('E', x)

def lift5():
    if False:
        for i in range(10):
            print('nop')
    print('A')

def liftcall1():
    if False:
        return 10
    x = 1
    print('A', x)
    with call_context:
        x += 1
    print('B', x)
    return x

def liftcall2():
    if False:
        while True:
            i = 10
    x = 1
    print('A', x)
    with call_context:
        x += 1
    print('B', x)
    with call_context:
        x += 10
    print('C', x)
    return x

def liftcall3():
    if False:
        i = 10
        return i + 15
    x = 1
    print('A', x)
    with call_context:
        if x > 0:
            x += 1
    print('B', x)
    with call_context:
        for i in range(10):
            x += i
    print('C', x)
    return x

def liftcall4():
    if False:
        for i in range(10):
            print('nop')
    with call_context:
        with call_context:
            pass

def liftcall5():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        with call_context:
            print(i)
            if i == 5:
                print('A')
                break
    return i

def lift_undefiend():
    if False:
        while True:
            i = 10
    with undefined_global_var:
        pass
bogus_contextmanager = object()

def lift_invalid():
    if False:
        while True:
            i = 10
    with bogus_contextmanager:
        pass
gv_type = types.intp

class TestWithFinding(TestCase):

    def check_num_of_with(self, func, expect_count):
        if False:
            for i in range(10):
                print('nop')
        the_ir = get_func_ir(func)
        ct = len(find_setupwiths(the_ir)[0])
        self.assertEqual(ct, expect_count)

    def test_lift1(self):
        if False:
            while True:
                i = 10
        self.check_num_of_with(lift1, expect_count=1)

    def test_lift2(self):
        if False:
            i = 10
            return i + 15
        self.check_num_of_with(lift2, expect_count=2)

    def test_lift3(self):
        if False:
            return 10
        self.check_num_of_with(lift3, expect_count=1)

    def test_lift4(self):
        if False:
            print('Hello World!')
        self.check_num_of_with(lift4, expect_count=2)

    def test_lift5(self):
        if False:
            while True:
                i = 10
        self.check_num_of_with(lift5, expect_count=0)

class BaseTestWithLifting(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(BaseTestWithLifting, self).setUp()
        self.typingctx = typing.Context()
        self.targetctx = cpu.CPUContext(self.typingctx)
        self.flags = DEFAULT_FLAGS

    def check_extracted_with(self, func, expect_count, expected_stdout):
        if False:
            return 10
        the_ir = get_func_ir(func)
        (new_ir, extracted) = with_lifting(the_ir, self.typingctx, self.targetctx, self.flags, locals={})
        self.assertEqual(len(extracted), expect_count)
        cres = self.compile_ir(new_ir)
        with captured_stdout() as out:
            cres.entry_point()
        self.assertEqual(out.getvalue(), expected_stdout)

    def compile_ir(self, the_ir, args=(), return_type=None):
        if False:
            return 10
        typingctx = self.typingctx
        targetctx = self.targetctx
        flags = self.flags
        with cpu_target.nested_context(typingctx, targetctx):
            return compile_ir(typingctx, targetctx, the_ir, args, return_type, flags, locals={})

class TestLiftByPass(BaseTestWithLifting):

    def test_lift1(self):
        if False:
            while True:
                i = 10
        self.check_extracted_with(lift1, expect_count=1, expected_stdout='A\nC\n')

    def test_lift2(self):
        if False:
            return 10
        self.check_extracted_with(lift2, expect_count=2, expected_stdout='A 1\nD 3\n')

    def test_lift3(self):
        if False:
            while True:
                i = 10
        self.check_extracted_with(lift3, expect_count=1, expected_stdout='A 1 100\nD 2 101\n')

    def test_lift4(self):
        if False:
            while True:
                i = 10
        self.check_extracted_with(lift4, expect_count=2, expected_stdout='A 0\nE 11\n')

    def test_lift5(self):
        if False:
            print('Hello World!')
        self.check_extracted_with(lift5, expect_count=0, expected_stdout='A\n')

class TestLiftCall(BaseTestWithLifting):

    def check_same_semantic(self, func):
        if False:
            i = 10
            return i + 15
        'Ensure same semantic with non-jitted code\n        '
        jitted = njit(func)
        with captured_stdout() as got:
            jitted()
        with captured_stdout() as expect:
            func()
        self.assertEqual(got.getvalue(), expect.getvalue())

    def test_liftcall1(self):
        if False:
            while True:
                i = 10
        self.check_extracted_with(liftcall1, expect_count=1, expected_stdout='A 1\nB 2\n')
        self.check_same_semantic(liftcall1)

    def test_liftcall2(self):
        if False:
            return 10
        self.check_extracted_with(liftcall2, expect_count=2, expected_stdout='A 1\nB 2\nC 12\n')
        self.check_same_semantic(liftcall2)

    def test_liftcall3(self):
        if False:
            return 10
        self.check_extracted_with(liftcall3, expect_count=2, expected_stdout='A 1\nB 2\nC 47\n')
        self.check_same_semantic(liftcall3)

    def test_liftcall4(self):
        if False:
            while True:
                i = 10
        accept = (errors.TypingError, errors.NumbaRuntimeError, errors.NumbaValueError, errors.CompilerError)
        with self.assertRaises(accept) as raises:
            njit(liftcall4)()
        msg = 'compiler re-entrant to the same function signature'
        self.assertIn(msg, str(raises.exception))

    @unittest.skipIf(PYVERSION <= (3, 8), 'unsupported on py3.8 and before')
    @expected_failure_py311
    def test_liftcall5(self):
        if False:
            print('Hello World!')
        self.check_extracted_with(liftcall5, expect_count=1, expected_stdout='0\n1\n2\n3\n4\n5\nA\n')
        self.check_same_semantic(liftcall5)

def expected_failure_for_list_arg(fn):
    if False:
        i = 10
        return i + 15

    def core(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        with self.assertRaises(errors.TypingError) as raises:
            fn(self, *args, **kwargs)
        self.assertIn('Does not support list type', str(raises.exception))
    return core

def expected_failure_for_function_arg(fn):
    if False:
        for i in range(10):
            print('nop')

    def core(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(errors.TypingError) as raises:
            fn(self, *args, **kwargs)
        self.assertIn('Does not support function type', str(raises.exception))
    return core

class TestLiftObj(MemoryLeak, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        warnings.simplefilter('error', errors.NumbaWarning)

    def tearDown(self):
        if False:
            while True:
                i = 10
        warnings.resetwarnings()

    def assert_equal_return_and_stdout(self, pyfunc, *args):
        if False:
            i = 10
            return i + 15
        py_args = copy.deepcopy(args)
        c_args = copy.deepcopy(args)
        cfunc = njit(pyfunc)
        with captured_stdout() as stream:
            expect_res = pyfunc(*py_args)
            expect_out = stream.getvalue()
        cfunc.compile(tuple(map(typeof, c_args)))
        with captured_stdout() as stream:
            got_res = cfunc(*c_args)
            got_out = stream.getvalue()
        self.assertEqual(expect_out, got_out)
        self.assertPreciseEqual(expect_res, got_res)

    def test_lift_objmode_basic(self):
        if False:
            while True:
                i = 10

        def bar(ival):
            if False:
                print('Hello World!')
            print('ival =', {'ival': ival // 2})

        def foo(ival):
            if False:
                while True:
                    i = 10
            ival += 1
            with objmode_context:
                bar(ival)
            return ival + 1

        def foo_nonglobal(ival):
            if False:
                while True:
                    i = 10
            ival += 1
            with numba.objmode:
                bar(ival)
            return ival + 1
        self.assert_equal_return_and_stdout(foo, 123)
        self.assert_equal_return_and_stdout(foo_nonglobal, 123)

    def test_lift_objmode_array_in(self):
        if False:
            return 10

        def bar(arr):
            if False:
                for i in range(10):
                    print('nop')
            print({'arr': arr // 2})
            arr *= 2

        def foo(nelem):
            if False:
                i = 10
                return i + 15
            arr = np.arange(nelem).astype(np.int64)
            with objmode_context:
                bar(arr)
            return arr + 1
        nelem = 10
        self.assert_equal_return_and_stdout(foo, nelem)

    def test_lift_objmode_define_new_unused(self):
        if False:
            i = 10
            return i + 15

        def bar(y):
            if False:
                return 10
            print(y)

        def foo(x):
            if False:
                while True:
                    i = 10
            with objmode_context():
                y = 2 + x
                a = np.arange(y)
                bar(a)
            return x
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_return_simple(self):
        if False:
            i = 10
            return i + 15

        def inverse(x):
            if False:
                for i in range(10):
                    print('nop')
            print(x)
            return 1 / x

        def foo(x):
            if False:
                print('Hello World!')
            with objmode_context(y='float64'):
                y = inverse(x)
            return (x, y)

        def foo_nonglobal(x):
            if False:
                for i in range(10):
                    print('nop')
            with numba.objmode(y='float64'):
                y = inverse(x)
            return (x, y)
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)
        self.assert_equal_return_and_stdout(foo_nonglobal, arg)

    def test_lift_objmode_return_array(self):
        if False:
            i = 10
            return i + 15

        def inverse(x):
            if False:
                for i in range(10):
                    print('nop')
            print(x)
            return 1 / x

        def foo(x):
            if False:
                print('Hello World!')
            with objmode_context(y='float64[:]', z='int64'):
                y = inverse(x)
                z = int(y[0])
            return (x, y, z)
        arg = np.arange(1, 10, dtype=np.float64)
        self.assert_equal_return_and_stdout(foo, arg)

    @expected_failure_for_list_arg
    def test_lift_objmode_using_list(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                return 10
            with objmode_context(y='float64[:]'):
                print(x)
                x[0] = 4
                print(x)
                y = [1, 2, 3] + x
                y = np.asarray([1 / i for i in y])
            return (x, y)
        arg = [1, 2, 3]
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_var_redef(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                print('Hello World!')
            for x in range(x):
                pass
            if x:
                x += 1
            with objmode_context(x='intp'):
                print(x)
                x -= 1
                print(x)
                for i in range(x):
                    x += i
                    print(x)
            return x
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    @expected_failure_for_list_arg
    def test_case01_mutate_list_ahead_of_ctx(self):
        if False:
            print('Hello World!')

        def foo(x, z):
            if False:
                return 10
            x[2] = z
            with objmode_context():
                print(x)
            with objmode_context():
                x[2] = 2 * z
                print(x)
            return x
        self.assert_equal_return_and_stdout(foo, [1, 2, 3], 15)

    def test_case02_mutate_array_ahead_of_ctx(self):
        if False:
            print('Hello World!')

        def foo(x, z):
            if False:
                for i in range(10):
                    print('nop')
            x[2] = z
            with objmode_context():
                print(x)
            with objmode_context():
                x[2] = 2 * z
                print(x)
            return x
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x, 15)

    @expected_failure_for_list_arg
    def test_case03_create_and_mutate(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                while True:
                    i = 10
            with objmode_context(y='List(int64)'):
                y = [1, 2, 3]
            with objmode_context():
                y[2] = 10
            return y
        self.assert_equal_return_and_stdout(foo, 1)

    def test_case04_bogus_variable_type_info(self):
        if False:
            return 10

        def foo(x):
            if False:
                return 10
            with objmode_context(k='float64[:]'):
                print(x)
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.TypingError) as raises:
            cfoo(x)
        self.assertIn('Invalid type annotation on non-outgoing variables', str(raises.exception))

    def test_case05_bogus_type_info(self):
        if False:
            i = 10
            return i + 15

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            with objmode_context(z='float64[:]'):
                z = x + 1j
            return z
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(TypeError) as raises:
            got = cfoo(x)
        self.assertIn("can't unbox array from PyObject into native value.  The object maybe of a different type", str(raises.exception))

    def test_case06_double_objmode(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            with objmode_context():
                with objmode_context():
                    print(x)
            return x
        with self.assertRaises(errors.TypingError) as raises:
            njit(foo)(123)
        pat = 'During: resolving callee type: type\\(ObjModeLiftedWith\\(<.*>\\)\\)'
        self.assertRegex(str(raises.exception), pat)

    def test_case07_mystery_key_error(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                return 10
            with objmode_context():
                t = {'a': x}
                u = 3
            return (x, t, u)
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.TypingError) as raises:
            cfoo(x)
        exstr = str(raises.exception)
        self.assertIn("Missing type annotation on outgoing variable(s): ['t', 'u']", exstr)
        self.assertIn("Example code: with objmode(t='<add_type_as_string_here>')", exstr)

    def test_case08_raise_from_external(self):
        if False:
            i = 10
            return i + 15
        d = dict()

        def foo(x):
            if False:
                return 10
            for i in range(len(x)):
                with objmode_context():
                    k = str(i)
                    v = x[i]
                    d[k] = v
                    print(d['2'])
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(KeyError) as raises:
            cfoo(x)
        self.assertEqual(str(raises.exception), "'2'")

    def test_case09_explicit_raise(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                i = 10
                return i + 15
            with objmode_context():
                raise ValueError()
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.CompilerError) as raises:
            cfoo(x)
        self.assertIn('unsupported control flow due to raise statements inside with block', str(raises.exception))

    @expected_failure_for_list_arg
    def test_case10_mutate_across_contexts(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                while True:
                    i = 10
            with objmode_context(y='List(int64)'):
                y = [1, 2, 3]
            with objmode_context():
                y[2] = 10
            return y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case10_mutate_array_across_contexts(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                i = 10
                return i + 15
            with objmode_context(y='int64[:]'):
                y = np.asarray([1, 2, 3], dtype='int64')
            with objmode_context():
                y[2] = 10
            return y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case11_define_function_in_context(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                return 10
            with objmode_context():

                def bar(y):
                    if False:
                        while True:
                            i = 10
                    return y + 1
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(NameError) as raises:
            cfoo(x)
        self.assertIn("global name 'bar' is not defined", str(raises.exception))

    def test_case12_njit_inside_a_objmode_ctx(self):
        if False:
            for i in range(10):
                print('nop')

        def bar(y):
            if False:
                while True:
                    i = 10
            return y + 1

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            with objmode_context(y='int64[:]'):
                y = njit(bar)(x).astype('int64')
            return x + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case14_return_direct_from_objmode_ctx(self):
        if False:
            return 10

        def foo(x):
            if False:
                print('Hello World!')
            with objmode_context(x='int64[:]'):
                x += 1
                return x
        if PYVERSION <= (3, 8):
            with self.assertRaises(errors.CompilerError) as raises:
                cfoo = njit(foo)
                cfoo(np.array([1, 2, 3]))
            msg = 'unsupported control flow: due to return statements inside with block'
            self.assertIn(msg, str(raises.exception))
        else:
            result = foo(np.array([1, 2, 3]))
            np.testing.assert_array_equal(np.array([2, 3, 4]), result)

    @unittest.expectedFailure
    def test_case15_close_over_objmode_ctx(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                while True:
                    i = 10
            j = 10

            def bar(x):
                if False:
                    print('Hello World!')
                with objmode_context(x='int64[:]'):
                    print(x)
                    return x + j
            return bar(x) + 2
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    @skip_unless_scipy
    def test_case16_scipy_call_in_objmode_ctx(self):
        if False:
            for i in range(10):
                print('nop')
        from scipy import sparse as sp

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            with objmode_context(k='int64'):
                print(x)
                spx = sp.csr_matrix(x)
                k = np.int64(spx[0, 0])
            return k
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case17_print_own_bytecode(self):
        if False:
            for i in range(10):
                print('nop')
        import dis

        def foo(x):
            if False:
                print('Hello World!')
            with objmode_context():
                dis.dis(foo)
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    @expected_failure_for_function_arg
    def test_case18_njitfunc_passed_to_objmode_ctx(self):
        if False:
            print('Hello World!')

        def foo(func, x):
            if False:
                print('Hello World!')
            with objmode_context():
                func(x[0])
        x = np.array([1, 2, 3])
        fn = njit(lambda z: z + 5)
        self.assert_equal_return_and_stdout(foo, fn, x)

    @expected_failure_py311
    def test_case19_recursion(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                print('Hello World!')
            with objmode_context():
                if x == 0:
                    return 7
            ret = foo(x - 1)
            return ret
        with self.assertRaises((errors.TypingError, errors.CompilerError)) as raises:
            cfoo = njit(foo)
            cfoo(np.array([1, 2, 3]))
        msg = "Untyped global name 'foo'"
        self.assertIn(msg, str(raises.exception))

    @unittest.expectedFailure
    def test_case20_rng_works_ok(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                i = 10
                return i + 15
            np.random.seed(0)
            y = np.random.rand()
            with objmode_context(z='float64'):
                z = np.random.rand()
            return x + z + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case21_rng_seed_works_ok(self):
        if False:
            return 10

        def foo(x):
            if False:
                while True:
                    i = 10
            np.random.seed(0)
            y = np.random.rand()
            with objmode_context(z='float64'):
                np.random.seed(0)
                z = np.random.rand()
            return x + z + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_example01(self):
        if False:
            while True:
                i = 10

        def bar(x):
            if False:
                return 10
            return np.asarray(list(reversed(x.tolist())))

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            x = np.arange(5)
            with objmode(y='intp[:]'):
                y = x + bar(x)
            return y
        self.assertPreciseEqual(foo(), foo.py_func())
        self.assertIs(objmode, objmode_context)

    def test_objmode_in_overload(self):
        if False:
            while True:
                i = 10

        def foo(s):
            if False:
                while True:
                    i = 10
            pass

        @overload(foo)
        def foo_overload(s):
            if False:
                return 10

            def impl(s):
                if False:
                    return 10
                with objmode(out='intp'):
                    out = s + 3
                return out
            return impl

        @numba.njit
        def f():
            if False:
                return 10
            return foo(1)
        self.assertEqual(f(), 1 + 3)

    def test_objmode_gv_variable(self):
        if False:
            while True:
                i = 10

        @njit
        def global_var():
            if False:
                for i in range(10):
                    print('nop')
            with objmode(val=gv_type):
                val = 12.3
            return val
        ret = global_var()
        self.assertIsInstance(ret, int)
        self.assertEqual(ret, 12)

    def test_objmode_gv_variable_error(self):
        if False:
            return 10

        @njit
        def global_var():
            if False:
                for i in range(10):
                    print('nop')
            with objmode(val=gv_type2):
                val = 123
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'val'. Global 'gv_type2' is not defined\\."):
            global_var()

    def test_objmode_gv_mod_attr(self):
        if False:
            i = 10
            return i + 15

        @njit
        def modattr1():
            if False:
                return 10
            with objmode(val=types.intp):
                val = 12.3
            return val

        @njit
        def modattr2():
            if False:
                return 10
            with objmode(val=numba.types.intp):
                val = 12.3
            return val
        for fn in (modattr1, modattr2):
            with self.subTest(fn=str(fn)):
                ret = fn()
                self.assertIsInstance(ret, int)
                self.assertEqual(ret, 12)

    def test_objmode_gv_mod_attr_error(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def moderror():
            if False:
                while True:
                    i = 10
            with objmode(val=types.THIS_DOES_NOT_EXIST):
                val = 12.3
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'val'. Getattr cannot be resolved at compile-time"):
            moderror()

    def test_objmode_gv_mod_attr_error_multiple(self):
        if False:
            print('Hello World!')

        @njit
        def moderror():
            if False:
                while True:
                    i = 10
            with objmode(v1=types.intp, v2=types.THIS_DOES_NOT_EXIST, v3=types.float32):
                v1 = 12.3
                v2 = 12.3
                v3 = 12.3
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'v2'. Getattr cannot be resolved at compile-time"):
            moderror()

    def test_objmode_closure_type_in_overload(self):
        if False:
            print('Hello World!')

        def foo():
            if False:
                print('Hello World!')
            pass

        @overload(foo)
        def foo_overload():
            if False:
                while True:
                    i = 10
            shrubbery = types.float64[:]

            def impl():
                if False:
                    return 10
                with objmode(out=shrubbery):
                    out = np.arange(10).astype(np.float64)
                return out
            return impl

        @njit
        def bar():
            if False:
                while True:
                    i = 10
            return foo()
        self.assertPreciseEqual(bar(), np.arange(10).astype(np.float64))

    def test_objmode_closure_type_in_overload_error(self):
        if False:
            for i in range(10):
                print('nop')

        def foo():
            if False:
                return 10
            pass

        @overload(foo)
        def foo_overload():
            if False:
                for i in range(10):
                    print('nop')
            shrubbery = types.float64[:]

            def impl():
                if False:
                    i = 10
                    return i + 15
                with objmode(out=shrubbery):
                    out = np.arange(10).astype(np.float64)
                return out
            del shrubbery
            return impl

        @njit
        def bar():
            if False:
                i = 10
                return i + 15
            return foo()
        with self.assertRaisesRegex(errors.TypingError, "Error handling objmode argument 'out'. Freevar 'shrubbery' is not defined"):
            bar()

    def test_objmode_invalid_use(self):
        if False:
            return 10

        @njit
        def moderror():
            if False:
                print('Hello World!')
            with objmode(bad=1 + 1):
                out = 1
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'bad'. The value must be a compile-time constant either as a non-local variable or a getattr expression that refers to a Numba type."):
            moderror()

    def test_objmode_multi_type_args(self):
        if False:
            print('Hello World!')
        array_ty = types.int32[:]

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            with objmode(t1='float64', t2=gv_type, t3=array_ty):
                t1 = 793856.5
                t2 = t1
                t3 = np.arange(5).astype(np.int32)
            return (t1, t2, t3)
        (t1, t2, t3) = foo()
        self.assertPreciseEqual(t1, 793856.5)
        self.assertPreciseEqual(t2, 793856)
        self.assertPreciseEqual(t3, np.arange(5).astype(np.int32))

    def test_objmode_jitclass(self):
        if False:
            while True:
                i = 10
        spec = [('value', types.int32), ('array', types.float32[:])]

        @jitclass(spec)
        class Bag(object):

            def __init__(self, value):
                if False:
                    while True:
                        i = 10
                self.value = value
                self.array = np.zeros(value, dtype=np.float32)

            @property
            def size(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.array.size

            def increment(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(self.size):
                    self.array[i] += val
                return self.array

            @staticmethod
            def add(x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y
        n = 21
        mybag = Bag(n)

        def foo():
            if False:
                i = 10
                return i + 15
            pass

        @overload(foo)
        def foo_overload():
            if False:
                return 10
            shrubbery = mybag._numba_type_

            def impl():
                if False:
                    return 10
                with objmode(out=shrubbery):
                    out = Bag(123)
                    out.increment(3)
                return out
            return impl

        @njit
        def bar():
            if False:
                return 10
            return foo()
        z = bar()
        self.assertIsInstance(z, Bag)
        self.assertEqual(z.add(2, 3), 2 + 3)
        exp_array = np.zeros(123, dtype=np.float32) + 3
        self.assertPreciseEqual(z.array, exp_array)

    @staticmethod
    def case_objmode_cache(x):
        if False:
            for i in range(10):
                print('nop')
        with objmode(output='float64'):
            output = x / 10
        return output

    def test_objmode_reflected_list(self):
        if False:
            for i in range(10):
                print('nop')
        ret_type = typeof([1, 2, 3, 4, 5])

        @njit
        def test2():
            if False:
                print('Hello World!')
            with objmode(out=ret_type):
                out = [1, 2, 3, 4, 5]
            return out
        with self.assertRaises(errors.CompilerError) as raises:
            test2()
        self.assertRegex(str(raises.exception), "Objmode context failed. Argument 'out' is declared as an unsupported type: reflected list\\(int(32|64)\\)<iv=None>. Reflected types are not supported.")

    def test_objmode_reflected_set(self):
        if False:
            while True:
                i = 10
        ret_type = typeof({1, 2, 3, 4, 5})

        @njit
        def test2():
            if False:
                for i in range(10):
                    print('nop')
            with objmode(result=ret_type):
                result = {1, 2, 3, 4, 5}
            return result
        with self.assertRaises(errors.CompilerError) as raises:
            test2()
        self.assertRegex(str(raises.exception), "Objmode context failed. Argument 'result' is declared as an unsupported type: reflected set\\(int(32|64)\\). Reflected types are not supported.")

    def test_objmode_typed_dict(self):
        if False:
            print('Hello World!')
        ret_type = types.DictType(types.unicode_type, types.int64)

        @njit
        def test4():
            if False:
                for i in range(10):
                    print('nop')
            with objmode(res=ret_type):
                res = {'A': 1, 'B': 2}
            return res
        with self.assertRaises(TypeError) as raises:
            test4()
        self.assertIn("can't unbox a <class 'dict'> as a <class 'numba.typed.typeddict.Dict'>", str(raises.exception))

    def test_objmode_typed_list(self):
        if False:
            for i in range(10):
                print('nop')
        ret_type = types.ListType(types.int64)

        @njit
        def test4():
            if False:
                return 10
            with objmode(res=ret_type):
                res = [1, 2]
            return res
        with self.assertRaises(TypeError) as raises:
            test4()
        self.assertRegex(str(raises.exception), "can't unbox a <class 'list'> as a (<class ')?numba.typed.typedlist.List('>)?")

    def test_objmode_use_of_view(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(x):
            if False:
                i = 10
                return i + 15
            with numba.objmode(y='int64[::1]'):
                y = x.view('int64')
            return y
        a = np.ones(1, np.int64).view('float64')
        expected = foo.py_func(a)
        got = foo(a)
        self.assertPreciseEqual(expected, got)

def case_inner_pyfunc(x):
    if False:
        return 10
    return x / 10

def case_objmode_cache(x):
    if False:
        print('Hello World!')
    with objmode(output='float64'):
        output = case_inner_pyfunc(x)
    return output

class TestLiftObjCaching(MemoryLeak, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        warnings.simplefilter('error', errors.NumbaWarning)

    def tearDown(self):
        if False:
            print('Hello World!')
        warnings.resetwarnings()

    def check(self, py_func):
        if False:
            while True:
                i = 10
        first = njit(cache=True)(py_func)
        self.assertEqual(first(123), 12.3)
        second = njit(cache=True)(py_func)
        self.assertFalse(second._cache_hits)
        self.assertEqual(second(123), 12.3)
        self.assertTrue(second._cache_hits)

    def test_objmode_caching_basic(self):
        if False:
            print('Hello World!')

        def pyfunc(x):
            if False:
                print('Hello World!')
            with objmode(output='float64'):
                output = x / 10
            return output
        self.check(pyfunc)

    def test_objmode_caching_call_closure_bad(self):
        if False:
            i = 10
            return i + 15

        def other_pyfunc(x):
            if False:
                for i in range(10):
                    print('nop')
            return x / 10

        def pyfunc(x):
            if False:
                return 10
            with objmode(output='float64'):
                output = other_pyfunc(x)
            return output
        self.check(pyfunc)

    def test_objmode_caching_call_closure_good(self):
        if False:
            while True:
                i = 10
        self.check(case_objmode_cache)

class TestBogusContext(BaseTestWithLifting):

    def test_undefined_global(self):
        if False:
            i = 10
            return i + 15
        the_ir = get_func_ir(lift_undefiend)
        with self.assertRaises(errors.CompilerError) as raises:
            with_lifting(the_ir, self.typingctx, self.targetctx, self.flags, locals={})
        self.assertIn('Undefined variable used as context manager', str(raises.exception))

    def test_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        the_ir = get_func_ir(lift_invalid)
        with self.assertRaises(errors.CompilerError) as raises:
            with_lifting(the_ir, self.typingctx, self.targetctx, self.flags, locals={})
        self.assertIn('Unsupported context manager in use', str(raises.exception))

    def test_with_as_fails_gracefully(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                print('Hello World!')
            with open('') as f:
                pass
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        excstr = str(raises.exception)
        msg = "The 'with (context manager) as (variable):' construct is not supported."
        self.assertIn(msg, excstr)

class TestMisc(TestCase):
    _numba_parallel_test_ = False

    @linux_only
    @TestCase.run_test_in_subprocess
    def test_no_fork_in_compilation(self):
        if False:
            while True:
                i = 10
        if not strace_supported():
            self.skipTest('strace support missing')

        def force_compile():
            if False:
                print('Hello World!')

            @njit('void()')
            def f():
                if False:
                    while True:
                        i = 10
                with numba.objmode():
                    pass
        syscalls = ['fork', 'clone', 'execve']
        strace_data = strace(force_compile, syscalls)
        self.assertFalse(strace_data)
if __name__ == '__main__':
    unittest.main()