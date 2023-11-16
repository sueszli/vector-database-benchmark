import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np

def dump(foo):
    if False:
        while True:
            i = 10
    from numba.core import function
    foo_type = function.fromobject(foo)
    foo_sig = foo_type.signature()
    foo.compile(foo_sig)
    print('{" LLVM IR OF "+foo.__name__+" ":*^70}')
    print(foo.inspect_llvm(foo_sig.args))
    print('{"":*^70}')

def mk_cfunc_func(sig):
    if False:
        return 10

    def cfunc_func(func):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(func, pytypes.FunctionType), repr(func)
        f = cfunc(sig)(func)
        f.pyfunc = func
        return f
    return cfunc_func

def njit_func(func):
    if False:
        print('Hello World!')
    assert isinstance(func, pytypes.FunctionType), repr(func)
    f = jit(nopython=True)(func)
    f.pyfunc = func
    return f

def mk_njit_with_sig_func(sig):
    if False:
        print('Hello World!')

    def njit_with_sig_func(func):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(func, pytypes.FunctionType), repr(func)
        f = jit(sig, nopython=True)(func)
        f.pyfunc = func
        return f
    return njit_with_sig_func

def mk_ctypes_func(sig):
    if False:
        while True:
            i = 10

    def ctypes_func(func, sig=int64(int64)):
        if False:
            print('Hello World!')
        assert isinstance(func, pytypes.FunctionType), repr(func)
        cfunc = mk_cfunc_func(sig)(func)
        addr = cfunc._wrapper_address
        if sig == int64(int64):
            f = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
            f.pyfunc = func
            return f
        raise NotImplementedError(f'ctypes decorator for {func} with signature {sig}')
    return ctypes_func

class WAP(types.WrapperAddressProtocol):
    """An example implementation of wrapper address protocol.

    """

    def __init__(self, func, sig):
        if False:
            while True:
                i = 10
        self.pyfunc = func
        self.cfunc = cfunc(sig)(func)
        self.sig = sig

    def __wrapper_address__(self):
        if False:
            while True:
                i = 10
        return self.cfunc._wrapper_address

    def signature(self):
        if False:
            while True:
                i = 10
        return self.sig

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        return self.pyfunc(*args, **kwargs)

def mk_wap_func(sig):
    if False:
        return 10

    def wap_func(func):
        if False:
            while True:
                i = 10
        return WAP(func, sig)
    return wap_func

class TestFunctionType(TestCase):
    """Test first-class functions in the context of a Numba jit compiled
    function.

    """

    def test_in__(self):
        if False:
            print('Hello World!')
        'Function is passed in as an argument.\n        '

        def a(i):
            if False:
                print('Hello World!')
            return i + 1

        def foo(f):
            if False:
                return 10
            return 0
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_ctypes_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__, jit=jit_opts):
                    a_ = decor(a)
                    self.assertEqual(jit_(foo)(a_), foo(a))

    def test_in_call__(self):
        if False:
            for i in range(10):
                print('nop')
        'Function is passed in as an argument and called.\n        Also test different return values.\n        '

        def a_i64(i):
            if False:
                return 10
            return i + 1234567

        def a_f64(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 1.5

        def a_str(i):
            if False:
                return 10
            return 'abc'

        def foo(f):
            if False:
                print('Hello World!')
            return f(123)
        for (f, sig) in [(a_i64, int64(int64)), (a_f64, float64(int64))]:
            for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
                for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                    jit_ = jit(**jit_opts)
                    with self.subTest(sig=sig, decor=decor.__name__, jit=jit_opts):
                        f_ = decor(f)
                        self.assertEqual(jit_(foo)(f_), foo(f))

    def test_in_call_out(self):
        if False:
            print('Hello World!')
        'Function is passed in as an argument, called, and returned.\n        '

        def a(i):
            if False:
                while True:
                    i = 10
            return i + 1

        def foo(f):
            if False:
                for i in range(10):
                    print('nop')
            f(123)
            return f
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    r1 = jit_(foo)(a_).pyfunc
                    r2 = foo(a)
                    self.assertEqual(r1, r2)

    def test_in_seq_call(self):
        if False:
            print('Hello World!')
        'Functions are passed in as arguments, used as tuple items, and\n        called.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def b(i):
            if False:
                i = 10
                return i + 15
            return i + 2

        def foo(f, g):
            if False:
                while True:
                    i = 10
            r = 0
            for f_ in (f, g):
                r = r + f_(r)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_), foo(a, b))

    def test_in_ns_seq_call(self):
        if False:
            for i in range(10):
                print('nop')
        'Functions are passed in as an argument and via namespace scoping\n        (mixed pathways), used as tuple items, and called.\n\n        '

        def a(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 1

        def b(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 2

        def mkfoo(b_):
            if False:
                print('Hello World!')

            def foo(f):
                if False:
                    for i in range(10):
                        print('nop')
                r = 0
                for f_ in (f, b_):
                    r = r + f_(r)
                return r
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(mkfoo(b_))(a_), mkfoo(b)(a))

    def test_ns_call(self):
        if False:
            return 10
        'Function is passed in via namespace scoping and called.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def mkfoo(a_):
            if False:
                while True:
                    i = 10

            def foo():
                if False:
                    return 10
                return a_(123)
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))(), mkfoo(a)())

    def test_ns_out(self):
        if False:
            for i in range(10):
                print('nop')
        'Function is passed in via namespace scoping and returned.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def mkfoo(a_):
            if False:
                while True:
                    i = 10

            def foo():
                if False:
                    return 10
                return a_
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_ns_call_out(self):
        if False:
            while True:
                i = 10
        'Function is passed in via namespace scoping, called, and then\n        returned.\n\n        '

        def a(i):
            if False:
                return 10
            return i + 1

        def mkfoo(a_):
            if False:
                for i in range(10):
                    print('nop')

            def foo():
                if False:
                    return 10
                a_(123)
                return a_
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(jit_(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_in_overload(self):
        if False:
            return 10
        'Function is passed in as an argument and called with different\n        argument types.\n\n        '

        def a(i):
            if False:
                return 10
            return i + 1

        def foo(f):
            if False:
                return 10
            r1 = f(123)
            r2 = f(123.45)
            return (r1, r2)
        for decor in [njit_func]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(foo)(a_), foo(a))

    def test_ns_overload(self):
        if False:
            i = 10
            return i + 15
        'Function is passed in via namespace scoping and called with\n        different argument types.\n\n        '

        def a(i):
            if False:
                print('Hello World!')
            return i + 1

        def mkfoo(a_):
            if False:
                for i in range(10):
                    print('nop')

            def foo():
                if False:
                    while True:
                        i = 10
                r1 = a_(123)
                r2 = a_(123.45)
                return (r1, r2)
            return foo
        for decor in [njit_func]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))(), mkfoo(a)())

    def test_in_choose(self):
        if False:
            for i in range(10):
                print('nop')
        'Functions are passed in as arguments and called conditionally.\n\n        '

        def a(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 1

        def b(i):
            if False:
                i = 10
                return i + 15
            return i + 2

        def foo(a, b, choose_left):
            if False:
                while True:
                    i = 10
            if choose_left:
                r = a(1)
            else:
                r = b(2)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True), foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False), foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True), foo(a, b, False))

    def test_ns_choose(self):
        if False:
            while True:
                i = 10
        'Functions are passed in via namespace scoping and called\n        conditionally.\n\n        '

        def a(i):
            if False:
                return 10
            return i + 1

        def b(i):
            if False:
                print('Hello World!')
            return i + 2

        def mkfoo(a_, b_):
            if False:
                for i in range(10):
                    print('nop')

            def foo(choose_left):
                if False:
                    return 10
                if choose_left:
                    r = a_(1)
                else:
                    r = b_(2)
                return r
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(mkfoo(a_, b_))(True), mkfoo(a, b)(True))
                    self.assertEqual(jit_(mkfoo(a_, b_))(False), mkfoo(a, b)(False))
                    self.assertNotEqual(jit_(mkfoo(a_, b_))(True), mkfoo(a, b)(False))

    def test_in_choose_out(self):
        if False:
            for i in range(10):
                print('nop')
        'Functions are passed in as arguments and returned conditionally.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def b(i):
            if False:
                print('Hello World!')
            return i + 2

        def foo(a, b, choose_left):
            if False:
                i = 10
                return i + 15
            if choose_left:
                return a
            else:
                return b
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False).pyfunc, foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, False))

    def test_in_choose_func_value(self):
        if False:
            return 10
        'Functions are passed in as arguments, selected conditionally and\n        called.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def b(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 2

        def foo(a, b, choose_left):
            if False:
                for i in range(10):
                    print('nop')
            if choose_left:
                f = a
            else:
                f = b
            return f(1)
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), njit_func, mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True), foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False), foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True), foo(a, b, False))

    def test_in_pick_func_call(self):
        if False:
            i = 10
            return i + 15
        'Functions are passed in as items of tuple argument, retrieved via\n        indexing, and called.\n\n        '

        def a(i):
            if False:
                print('Hello World!')
            return i + 1

        def b(i):
            if False:
                for i in range(10):
                    print('nop')
            return i + 2

        def foo(funcs, i):
            if False:
                print('Hello World!')
            f = funcs[i]
            r = f(123)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)((a_, b_), 0), foo((a, b), 0))
                    self.assertEqual(jit_(foo)((a_, b_), 1), foo((a, b), 1))
                    self.assertNotEqual(jit_(foo)((a_, b_), 0), foo((a, b), 1))

    def test_in_iter_func_call(self):
        if False:
            print('Hello World!')
        'Functions are passed in as items of tuple argument, retrieved via\n        indexing, and called within a variable for-loop.\n\n        '

        def a(i):
            if False:
                i = 10
                return i + 15
            return i + 1

        def b(i):
            if False:
                i = 10
                return i + 15
            return i + 2

        def foo(funcs, n):
            if False:
                return 10
            r = 0
            for i in range(n):
                f = funcs[i]
                r = r + f(r)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)((a_, b_), 2), foo((a, b), 2))

    def test_experimental_feature_warning(self):
        if False:
            for i in range(10):
                print('nop')

        @jit(nopython=True)
        def more(x):
            if False:
                print('Hello World!')
            return x + 1

        @jit(nopython=True)
        def less(x):
            if False:
                return 10
            return x - 1

        @jit(nopython=True)
        def foo(sel, x):
            if False:
                while True:
                    i = 10
            fn = more if sel else less
            return fn(x)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            res = foo(True, 10)
        self.assertEqual(res, 11)
        self.assertEqual(foo(False, 10), 9)
        self.assertGreaterEqual(len(ws), 1)
        pat = 'First-class function type feature is experimental'
        for w in ws:
            if pat in str(w.message):
                break
        else:
            self.fail('missing warning')

class TestFunctionTypeExtensions(TestCase):
    """Test calling external library functions within Numba jit compiled
    functions.

    """

    def test_wrapper_address_protocol_libm(self):
        if False:
            for i in range(10):
                print('nop')
        'Call cos and sinf from standard math library.\n\n        '
        import ctypes.util

        class LibM(types.WrapperAddressProtocol):

            def __init__(self, fname):
                if False:
                    return 10
                if IS_WIN32:
                    lib = ctypes.cdll.msvcrt
                else:
                    libpath = ctypes.util.find_library('m')
                    lib = ctypes.cdll.LoadLibrary(libpath)
                self.lib = lib
                self._name = fname
                if fname == 'cos':
                    addr = ctypes.cast(self.lib.cos, ctypes.c_voidp).value
                    signature = float64(float64)
                elif fname == 'sinf':
                    addr = ctypes.cast(self.lib.sinf, ctypes.c_voidp).value
                    signature = float32(float32)
                else:
                    raise NotImplementedError(f'wrapper address of `{fname}` with signature `{signature}`')
                self._signature = signature
                self._address = addr

            def __repr__(self):
                if False:
                    print('Hello World!')
                return f'{type(self).__name__}({self._name!r})'

            def __wrapper_address__(self):
                if False:
                    i = 10
                    return i + 15
                return self._address

            def signature(self):
                if False:
                    while True:
                        i = 10
                return self._signature
        mycos = LibM('cos')
        mysin = LibM('sinf')

        def myeval(f, x):
            if False:
                print('Hello World!')
            return f(x)
        for jit_opts in [dict(nopython=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(jit=jit_opts):
                if mycos.signature() is not None:
                    self.assertEqual(jit_(myeval)(mycos, 0.0), 1.0)
                if mysin.signature() is not None:
                    self.assertEqual(jit_(myeval)(mysin, float32(0.0)), 0.0)

    def test_compilation_results(self):
        if False:
            i = 10
            return i + 15
        'Turn the existing compilation results of a dispatcher instance to\n        first-class functions with precise types.\n        '

        @jit(nopython=True)
        def add_template(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y
        self.assertEqual(add_template(1, 2), 3)
        self.assertEqual(add_template(1.2, 3.4), 4.6)
        (cres1, cres2) = add_template.overloads.values()
        iadd = types.CompileResultWAP(cres1)
        fadd = types.CompileResultWAP(cres2)

        @jit(nopython=True)
        def foo(add, x, y):
            if False:
                i = 10
                return i + 15
            return add(x, y)

        @jit(forceobj=True)
        def foo_obj(add, x, y):
            if False:
                i = 10
                return i + 15
            return add(x, y)
        self.assertEqual(foo(iadd, 3, 4), 7)
        self.assertEqual(foo(fadd, 3.4, 4.5), 7.9)
        self.assertEqual(foo_obj(iadd, 3, 4), 7)
        self.assertEqual(foo_obj(fadd, 3.4, 4.5), 7.9)

class TestMiscIssues(TestCase):
    """Test issues of using first-class functions in the context of Numba
    jit compiled functions.

    """

    def test_issue_3405_using_cfunc(self):
        if False:
            while True:
                i = 10

        @cfunc('int64()')
        def a():
            if False:
                while True:
                    i = 10
            return 2

        @cfunc('int64()')
        def b():
            if False:
                i = 10
                return i + 15
            return 3

        def g(arg):
            if False:
                return 10
            if arg:
                f = a
            else:
                f = b
            return f()
        self.assertEqual(jit(nopython=True)(g)(True), 2)
        self.assertEqual(jit(nopython=True)(g)(False), 3)

    def test_issue_3405_using_njit(self):
        if False:
            while True:
                i = 10

        @jit(nopython=True)
        def a():
            if False:
                return 10
            return 2

        @jit(nopython=True)
        def b():
            if False:
                return 10
            return 3

        def g(arg):
            if False:
                for i in range(10):
                    print('nop')
            if not arg:
                f = b
            else:
                f = a
            return f()
        self.assertEqual(jit(nopython=True)(g)(True), 2)
        self.assertEqual(jit(nopython=True)(g)(False), 3)

    def test_pr4967_example(self):
        if False:
            print('Hello World!')

        @cfunc('int64(int64)')
        def a(i):
            if False:
                while True:
                    i = 10
            return i + 1

        @cfunc('int64(int64)')
        def b(i):
            if False:
                return 10
            return i + 2

        @jit(nopython=True)
        def foo(f, g):
            if False:
                for i in range(10):
                    print('nop')
            i = f(2)
            seq = (f, g)
            for fun in seq:
                i += fun(i)
            return i
        a_ = a._pyfunc
        b_ = b._pyfunc
        self.assertEqual(foo(a, b), a_(2) + a_(a_(2)) + b_(a_(2) + a_(a_(2))))

    def test_pr4967_array(self):
        if False:
            print('Hello World!')
        import numpy as np

        @cfunc('intp(intp[:], float64[:])')
        def foo1(x, y):
            if False:
                while True:
                    i = 10
            return x[0] + y[0]

        @cfunc('intp(intp[:], float64[:])')
        def foo2(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x[0] - y[0]

        def bar(fx, fy, i):
            if False:
                return 10
            a = np.array([10], dtype=np.intp)
            b = np.array([12], dtype=np.float64)
            if i == 0:
                f = fx
            elif i == 1:
                f = fy
            else:
                return
            return f(a, b)
        r = jit(nopython=True, no_cfunc_wrapper=True)(bar)(foo1, foo2, 0)
        self.assertEqual(r, bar(foo1, foo2, 0))
        self.assertNotEqual(r, bar(foo1, foo2, 1))

    def test_reference_example(self):
        if False:
            i = 10
            return i + 15
        import numba

        @numba.njit
        def composition(funcs, x):
            if False:
                for i in range(10):
                    print('nop')
            r = x
            for f in funcs[::-1]:
                r = f(r)
            return r

        @numba.cfunc('double(double)')
        def a(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1.0

        @numba.njit()
        def b(x):
            if False:
                return 10
            return x * x
        r = composition((a, b, b, a), 0.5)
        self.assertEqual(r, (0.5 + 1.0) ** 4 + 1.0)
        r = composition((b, a, b, b, a), 0.5)
        self.assertEqual(r, ((0.5 + 1.0) ** 4 + 1.0) ** 2)

    def test_apply_function_in_function(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(f, f_inner):
            if False:
                while True:
                    i = 10
            return f(f_inner)

        @cfunc('int64(float64)')
        def f_inner(i):
            if False:
                print('Hello World!')
            return int64(i * 3)

        @cfunc(int64(types.FunctionType(f_inner._sig)))
        def f(f_inner):
            if False:
                i = 10
                return i + 15
            return f_inner(123.4)
        self.assertEqual(jit(nopython=True)(foo)(f, f_inner), foo(f._pyfunc, f_inner._pyfunc))

    def test_function_with_none_argument(self):
        if False:
            for i in range(10):
                print('nop')

        @cfunc(int64(types.none))
        def a(i):
            if False:
                i = 10
                return i + 15
            return 1

        @jit(nopython=True)
        def foo(f):
            if False:
                return 10
            return f(None)
        self.assertEqual(foo(a), 1)

    def test_constant_functions(self):
        if False:
            i = 10
            return i + 15

        @jit(nopython=True)
        def a():
            if False:
                print('Hello World!')
            return 123

        @jit(nopython=True)
        def b():
            if False:
                i = 10
                return i + 15
            return 456

        @jit(nopython=True)
        def foo():
            if False:
                while True:
                    i = 10
            return a() + b()
        r = foo()
        if r != 123 + 456:
            print(foo.overloads[()].library.get_llvm_str())
        self.assertEqual(r, 123 + 456)

    def test_generators(self):
        if False:
            print('Hello World!')

        @jit(forceobj=True)
        def gen(xs):
            if False:
                return 10
            for x in xs:
                x += 1
                yield x

        @jit(forceobj=True)
        def con(gen_fn, xs):
            if False:
                return 10
            return [it for it in gen_fn(xs)]
        self.assertEqual(con(gen, (1, 2, 3)), [2, 3, 4])

        @jit(nopython=True)
        def gen_(xs):
            if False:
                return 10
            for x in xs:
                x += 1
                yield x
        self.assertEqual(con(gen_, (1, 2, 3)), [2, 3, 4])

    def test_jit_support(self):
        if False:
            print('Hello World!')

        @jit(nopython=True)
        def foo(f, x):
            if False:
                while True:
                    i = 10
            return f(x)

        @jit()
        def a(x):
            if False:
                i = 10
                return i + 15
            return x + 1

        @jit()
        def a2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x - 1

        @jit()
        def b(x):
            if False:
                i = 10
                return i + 15
            return x + 1.5
        self.assertEqual(foo(a, 1), 2)
        a2(5)
        self.assertEqual(foo(a2, 2), 1)
        self.assertEqual(foo(a2, 3), 2)
        self.assertEqual(foo(a, 2), 3)
        self.assertEqual(foo(a, 1.5), 2.5)
        self.assertEqual(foo(a2, 1), 0)
        self.assertEqual(foo(a, 2.5), 3.5)
        self.assertEqual(foo(b, 1.5), 3.0)
        self.assertEqual(foo(b, 1), 2.5)

    def test_signature_mismatch(self):
        if False:
            for i in range(10):
                print('nop')

        @jit(nopython=True)
        def f1(x):
            if False:
                i = 10
                return i + 15
            return x

        @jit(nopython=True)
        def f2(x):
            if False:
                i = 10
                return i + 15
            return x

        @jit(nopython=True)
        def foo(disp1, disp2, sel):
            if False:
                print('Hello World!')
            if sel == 1:
                fn = disp1
            else:
                fn = disp2
            return (fn([1]), fn(2))
        with self.assertRaises(errors.UnsupportedError) as cm:
            foo(f1, f2, sel=1)
        self.assertRegex(str(cm.exception), 'mismatch of function types:')
        self.assertEqual(foo(f1, f1, sel=1), ([1], 2))

    def test_unique_dispatcher(self):
        if False:
            for i in range(10):
                print('nop')

        def foo_template(funcs, x):
            if False:
                return 10
            r = x
            for f in funcs:
                r = f(r)
            return r
        a = jit(nopython=True)(lambda x: x + 1)
        b = jit(nopython=True)(lambda x: x + 2)
        foo = jit(nopython=True)(foo_template)
        a(0)
        a.disable_compile()
        r = foo((a, b), 0)
        self.assertEqual(r, 3)
        self.assertEqual(foo.signatures[0][0].dtype.is_precise(), True)

    def test_zero_address(self):
        if False:
            print('Hello World!')
        sig = int64()

        @cfunc(sig)
        def test():
            if False:
                while True:
                    i = 10
            return 123

        class Good(types.WrapperAddressProtocol):
            """A first-class function type with valid address.
            """

            def __wrapper_address__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return test.address

            def signature(self):
                if False:
                    while True:
                        i = 10
                return sig

        class Bad(types.WrapperAddressProtocol):
            """A first-class function type with invalid 0 address.
            """

            def __wrapper_address__(self):
                if False:
                    print('Hello World!')
                return 0

            def signature(self):
                if False:
                    i = 10
                    return i + 15
                return sig

        class BadToGood(types.WrapperAddressProtocol):
            """A first-class function type with invalid address that is
            recovered to a valid address.
            """
            counter = -1

            def __wrapper_address__(self):
                if False:
                    print('Hello World!')
                self.counter += 1
                return test.address * min(1, self.counter)

            def signature(self):
                if False:
                    return 10
                return sig
        good = Good()
        bad = Bad()
        bad2good = BadToGood()

        @jit(int64(sig.as_type()))
        def foo(func):
            if False:
                return 10
            return func()

        @jit(int64())
        def foo_good():
            if False:
                return 10
            return good()

        @jit(int64())
        def foo_bad():
            if False:
                print('Hello World!')
            return bad()

        @jit(int64())
        def foo_bad2good():
            if False:
                i = 10
                return i + 15
            return bad2good()
        self.assertEqual(foo(good), 123)
        self.assertEqual(foo_good(), 123)
        with self.assertRaises(ValueError) as cm:
            foo(bad)
        self.assertRegex(str(cm.exception), 'wrapper address of <.*> instance must be a positive')
        with self.assertRaises(RuntimeError) as cm:
            foo_bad()
        self.assertRegex(str(cm.exception), '.* function address is null')
        self.assertEqual(foo_bad2good(), 123)

    def test_issue_5470(self):
        if False:
            print('Hello World!')

        @njit()
        def foo1():
            if False:
                print('Hello World!')
            return 10

        @njit()
        def foo2():
            if False:
                while True:
                    i = 10
            return 20
        formulae_foo = (foo1, foo1)

        @njit()
        def bar_scalar(f1, f2):
            if False:
                return 10
            return f1() + f2()

        @njit()
        def bar():
            if False:
                print('Hello World!')
            return bar_scalar(*formulae_foo)
        self.assertEqual(bar(), 20)
        formulae_foo = (foo1, foo2)

        @njit()
        def bar():
            if False:
                print('Hello World!')
            return bar_scalar(*formulae_foo)
        self.assertEqual(bar(), 30)

    def test_issue_5540(self):
        if False:
            return 10

        @njit(types.int64(types.int64))
        def foo(x):
            if False:
                print('Hello World!')
            return x + 1

        @njit
        def bar_bad(foos):
            if False:
                for i in range(10):
                    print('nop')
            f = foos[0]
            return f(x=1)

        @njit
        def bar_good(foos):
            if False:
                return 10
            f = foos[0]
            return f(1)
        self.assertEqual(bar_good((foo,)), 2)
        with self.assertRaises((errors.UnsupportedError, errors.TypingError)) as cm:
            bar_bad((foo,))
        self.assertRegex(str(cm.exception), '.*first-class function call cannot use keyword arguments')

    def test_issue_5615(self):
        if False:
            return 10

        @njit
        def foo1(x):
            if False:
                i = 10
                return i + 15
            return x + 1

        @njit
        def foo2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 2

        @njit
        def bar(fcs):
            if False:
                while True:
                    i = 10
            x = 0
            a = 10
            (i, j) = fcs[0]
            x += i(j(a))
            for t in literal_unroll(fcs):
                (i, j) = t
                x += i(j(a))
            return x
        tup = ((foo1, foo2), (foo2, foo1))
        self.assertEqual(bar(tup), 39)

    def test_issue_5685(self):
        if False:
            print('Hello World!')

        @njit
        def foo1():
            if False:
                while True:
                    i = 10
            return 1

        @njit
        def foo2(x):
            if False:
                while True:
                    i = 10
            return x + 1

        @njit
        def foo3(x):
            if False:
                return 10
            return x + 2

        @njit
        def bar(fcs):
            if False:
                print('Hello World!')
            r = 0
            for pair in literal_unroll(fcs):
                (f1, f2) = pair
                r += f1() + f2(2)
            return r
        self.assertEqual(bar(((foo1, foo2),)), 4)
        self.assertEqual(bar(((foo1, foo2), (foo1, foo3))), 9)

class TestBasicSubtyping(TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that a dispatcher object *with* a pre-compiled overload\n        can be used as input to another function with locked-down signature\n        '
        a = 1

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            return x + 1
        foo(a)
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            if False:
                while True:
                    i = 10
            return fc(a)
        self.assertEqual(bar(foo), foo(a))

    def test_basic2(self):
        if False:
            print('Hello World!')
        '\n        Test that a dispatcher object *without* a pre-compiled overload\n        can be used as input to another function with locked-down signature\n        '
        a = 1

        @njit
        def foo(x):
            if False:
                return 10
            return x + 1
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            if False:
                print('Hello World!')
            return fc(a)
        self.assertEqual(bar(foo), foo(a))

    def test_basic3(self):
        if False:
            return 10
        '\n        Test that a dispatcher object *without* a pre-compiled overload\n        can be used as input to another function with locked-down signature and\n        that it behaves as a truly generic function (foo1 does not get locked)\n        '
        a = 1

        @njit
        def foo1(x):
            if False:
                i = 10
                return i + 15
            return x + 1

        @njit
        def foo2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 2
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(int_int_fc))
        def bar(fc):
            if False:
                while True:
                    i = 10
            return fc(a)
        self.assertEqual(bar(foo1) + 1, bar(foo2))

    def test_basic4(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that a dispatcher object can be used as input to another\n         function with signature as part of a tuple\n        '
        a = 1

        @njit
        def foo1(x):
            if False:
                print('Hello World!')
            return x + 1

        @njit
        def foo2(x):
            if False:
                print('Hello World!')
            return x + 2
        tup = (foo1, foo2)
        int_int_fc = types.FunctionType(types.int64(types.int64))

        @njit(types.int64(types.UniTuple(int_int_fc, 2)))
        def bar(fcs):
            if False:
                while True:
                    i = 10
            x = 0
            for i in range(2):
                x += fcs[i](a)
            return x
        self.assertEqual(bar(tup), foo1(a) + foo2(a))

    def test_basic5(self):
        if False:
            for i in range(10):
                print('nop')
        a = 1

        @njit
        def foo1(x):
            if False:
                i = 10
                return i + 15
            return x + 1

        @njit
        def foo2(x):
            if False:
                while True:
                    i = 10
            return x + 2

        @njit
        def bar1(x):
            if False:
                for i in range(10):
                    print('nop')
            return x / 10

        @njit
        def bar2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x / 1000
        tup = (foo1, foo2)
        tup_bar = (bar1, bar2)
        int_int_fc = types.FunctionType(types.int64(types.int64))
        flt_flt_fc = types.FunctionType(types.float64(types.float64))

        @njit((types.UniTuple(int_int_fc, 2), types.UniTuple(flt_flt_fc, 2)))
        def bar(fcs, ffs):
            if False:
                for i in range(10):
                    print('nop')
            x = 0
            for i in range(2):
                x += fcs[i](a)
            for fn in ffs:
                x += fn(a)
            return x
        got = bar(tup, tup_bar)
        expected = foo1(a) + foo2(a) + bar1(a) + bar2(a)
        self.assertEqual(got, expected)

class TestMultiFunctionType(MemoryLeakMixin, TestCase):

    def test_base(self):
        if False:
            return 10
        nb_array = typeof(np.ones(2))
        callee_int_type = types.FunctionType(int64(int64))
        sig_int = int64(callee_int_type, int64)
        callee_array_type = types.FunctionType(float64(nb_array))
        sig_array = float64(callee_array_type, nb_array)

        @njit([sig_int, sig_array])
        def caller(callee, a):
            if False:
                return 10
            return callee(a)

        @njit
        def callee_int(b):
            if False:
                while True:
                    i = 10
            return b

        @njit
        def callee_array(c):
            if False:
                i = 10
                return i + 15
            return c.sum()
        b = 1
        c = np.ones(2)
        self.assertEqual(caller(callee_int, b), b)
        self.assertEqual(caller(callee_array, c), c.sum())
if __name__ == '__main__':
    unittest.main()