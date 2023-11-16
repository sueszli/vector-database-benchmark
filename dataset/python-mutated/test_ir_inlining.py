"""
This tests the inline kwarg to @jit and @overload etc, it has nothing to do with
LLVM or low level inlining.
"""
import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import overload, overload_method, overload_attribute, register_model, models, make_attribute_wrapper, intrinsic, register_jitable
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import TestCase, unittest, MemoryLeakMixin, IRPreservingTestPipeline, skip_parfors_unsupported, ignore_internal_warnings
_GLOBAL1 = -50

@njit(inline='always')
def _global_func(x):
    if False:
        return 10
    return x + 1

def _global_defn(x):
    if False:
        for i in range(10):
            print('nop')
    return x + 1

@overload(_global_defn, inline='always')
def _global_overload(x):
    if False:
        while True:
            i = 10
    return _global_defn

class InliningBase(TestCase):
    _DEBUG = False
    inline_opt_as_bool = {'always': True, 'never': False}

    def sentinel_17_cost_model(self, func_ir):
        if False:
            return 10
        for blk in func_ir.blocks.values():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.FreeVar):
                        if stmt.value.value == 17:
                            return True
        return False

    def check(self, test_impl, *args, **kwargs):
        if False:
            return 10
        inline_expect = kwargs.pop('inline_expect', None)
        assert inline_expect
        block_count = kwargs.pop('block_count', 1)
        assert not kwargs
        for (k, v) in inline_expect.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(test_impl(*args), j_func(*args))
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = ir_utils.simplify_CFG(fir.blocks)
        if self._DEBUG:
            print('FIR'.center(80, '-'))
            fir.dump()
        if block_count != 'SKIP':
            self.assertEqual(len(fir.blocks), block_count)
        block = next(iter(fir.blocks.values()))
        exprs = [x for x in block.find_exprs()]
        assert exprs
        for (k, v) in inline_expect.items():
            found = False
            for expr in exprs:
                if getattr(expr, 'op', False) == 'call':
                    func_defn = fir.get_definition(expr.func)
                    found |= func_defn.name == k
                elif ir_utils.is_operator_or_getitem(expr):
                    found |= expr.fn.__name__ == k
            self.assertFalse(found == v)
        return fir
_GLOBAL = 1234

def _gen_involved():
    if False:
        i = 10
        return i + 15
    _FREEVAR = 51966

    def foo(a, b, c=12, d=1j, e=None):
        if False:
            for i in range(10):
                print('nop')
        f = a + b
        a += _FREEVAR
        g = np.zeros(c, dtype=np.complex64)
        h = f + g
        i = 1j / d
        n = 0
        t = 0
        if np.abs(i) > 0:
            k = h / i
            l = np.arange(1, c + 1)
            m = np.sqrt(l - g) + e * k
            if np.abs(m[0]) < 1:
                for o in range(a):
                    n += 0
                    if np.abs(n) < 3:
                        break
                n += m[2]
            p = g / l
            q = []
            for r in range(len(p)):
                q.append(p[r])
                if r > 4 + 1:
                    s = 123
                    t = 5
                    if s > 122 - c:
                        t += s
                t += q[0] + _GLOBAL
        return f + o + r + t + r + a + n
    return foo

class TestFunctionInlining(MemoryLeakMixin, InliningBase):

    def test_basic_inline_never(self):
        if False:
            print('Hello World!')

        @njit(inline='never')
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return

        def impl():
            if False:
                i = 10
                return i + 15
            return foo()
        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):
        if False:
            for i in range(10):
                print('nop')

        @njit(inline='always')
        def foo():
            if False:
                print('Hello World!')
            return

        def impl():
            if False:
                while True:
                    i = 10
            return foo()
        self.check(impl, inline_expect={'foo': True})

    def test_basic_inline_combos(self):
        if False:
            i = 10
            return i + 15

        def impl():
            if False:
                for i in range(10):
                    print('nop')
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for (inline_foo, inline_bar, inline_baz) in product(opts, opts, opts):

            @njit(inline=inline_foo)
            def foo():
                if False:
                    print('Hello World!')
                return

            @njit(inline=inline_bar)
            def bar():
                if False:
                    for i in range(10):
                        print('nop')
                return

            @njit(inline=inline_baz)
            def baz():
                if False:
                    print('Hello World!')
                return
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    @unittest.skip('Need to work out how to prevent this')
    def test_recursive_inline(self):
        if False:
            print('Hello World!')

        @njit(inline='always')
        def foo(x):
            if False:
                i = 10
                return i + 15
            if x == 0:
                return 12
            else:
                foo(x - 1)
        a = 3

        def impl():
            if False:
                print('Hello World!')
            b = 0
            if a > 1:
                b += 1
            foo(5)
            if b < a:
                b -= 1
        self.check(impl, inline_expect={'foo': True})

    def test_freevar_bindings(self):
        if False:
            while True:
                i = 10

        def factory(inline, x, y):
            if False:
                return 10
            z = x + 12

            @njit(inline=inline)
            def func():
                if False:
                    for i in range(10):
                        print('nop')
                return (x, y + 3, z)
            return func

        def impl():
            if False:
                i = 10
                return i + 15
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for (inline_foo, inline_bar, inline_baz) in product(opts, opts, opts):
            foo = factory(inline_foo, 10, 20)
            bar = factory(inline_bar, 30, 40)
            baz = factory(inline_baz, 50, 60)
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_global_binding(self):
        if False:
            for i in range(10):
                print('nop')

        def impl():
            if False:
                while True:
                    i = 10
            x = 19
            return _global_func(x)
        self.check(impl, inline_expect={'_global_func': True})

    def test_inline_from_another_module(self):
        if False:
            i = 10
            return i + 15
        from .inlining_usecases import bar

        def impl():
            if False:
                while True:
                    i = 10
            z = _GLOBAL1 + 2
            return (bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_getattr(self):
        if False:
            return 10
        import numba.tests.inlining_usecases as iuc

        def impl():
            if False:
                i = 10
                return i + 15
            z = _GLOBAL1 + 2
            return (iuc.bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_w_2_getattr(self):
        if False:
            return 10
        import numba.tests.inlining_usecases
        import numba.tests as nt

        def impl():
            if False:
                print('Hello World!')
            z = _GLOBAL1 + 2
            return (nt.inlining_usecases.bar(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_from_another_module_as_freevar(self):
        if False:
            while True:
                i = 10

        def factory():
            if False:
                return 10
            from .inlining_usecases import bar

            @njit(inline='always')
            def tmp():
                if False:
                    i = 10
                    return i + 15
                return bar()
            return tmp
        baz = factory()

        def impl():
            if False:
                while True:
                    i = 10
            z = _GLOBAL1 + 2
            return (baz(), z)
        self.check(impl, inline_expect={'bar': True})

    def test_inline_w_freevar_from_another_module(self):
        if False:
            return 10
        from .inlining_usecases import baz_factory

        def gen(a, b):
            if False:
                i = 10
                return i + 15
            bar = baz_factory(a)

            def impl():
                if False:
                    i = 10
                    return i + 15
                z = _GLOBAL1 + a * b
                return (bar(), z, a)
            return impl
        impl = gen(10, 20)
        self.check(impl, inline_expect={'bar': True})

    def test_inlining_models(self):
        if False:
            return 10

        def s17_caller_model(expr, caller_info, callee_info):
            if False:
                return 10
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(caller_info)

        def s17_callee_model(expr, caller_info, callee_info):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(callee_info)
        for (caller, callee) in ((11, 17), (17, 11)):

            @njit(inline=s17_caller_model)
            def foo():
                if False:
                    return 10
                return callee

            def impl(z):
                if False:
                    for i in range(10):
                        print('nop')
                x = z + caller
                y = foo()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'foo': caller == 17})
        for (caller, callee) in ((11, 17), (17, 11)):

            @njit(inline=s17_callee_model)
            def bar():
                if False:
                    i = 10
                    return i + 15
                return callee

            def impl(z):
                if False:
                    while True:
                        i = 10
                x = z + caller
                y = bar()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_inline_inside_loop(self):
        if False:
            print('Hello World!')

        @njit(inline='always')
        def foo():
            if False:
                while True:
                    i = 10
            return 12

        def impl():
            if False:
                for i in range(10):
                    print('nop')
            acc = 0.0
            for i in range(5):
                acc += foo()
            return acc
        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_inside_closure_inside_loop(self):
        if False:
            while True:
                i = 10

        @njit(inline='always')
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return 12

        def impl():
            if False:
                print('Hello World!')
            acc = 0.0
            for i in range(5):

                def bar():
                    if False:
                        i = 10
                        return i + 15
                    return foo() + 7
                acc += bar()
            return acc
        self.check(impl, inline_expect={'foo': True}, block_count=4)

    def test_inline_closure_inside_inlinable_inside_closure(self):
        if False:
            for i in range(10):
                print('nop')

        @njit(inline='always')
        def foo(a):
            if False:
                while True:
                    i = 10

            def baz():
                if False:
                    for i in range(10):
                        print('nop')
                return 12 + a
            return baz() + 8

        def impl():
            if False:
                return 10
            z = 9

            def bar(x):
                if False:
                    while True:
                        i = 10
                return foo(z) + 7 + x
            return bar(z + 2)
        self.check(impl, inline_expect={'foo': True}, block_count=1)

    def test_inline_involved(self):
        if False:
            while True:
                i = 10
        fortran = njit(inline='always')(_gen_involved())

        @njit(inline='always')
        def boz(j):
            if False:
                while True:
                    i = 10
            acc = 0

            def biz(t):
                if False:
                    i = 10
                    return i + 15
                return t + acc
            for x in range(j):
                acc += biz(8 + acc) + fortran(2.0, acc, 1, 12j, biz(acc))
            return acc

        @njit(inline='always')
        def foo(a):
            if False:
                i = 10
                return i + 15
            acc = 0
            for p in range(12):
                tmp = fortran(1, 1, 1, 1, 1)

                def baz(x):
                    if False:
                        i = 10
                        return i + 15
                    return 12 + a + x + tmp
                acc += baz(p) + 8 + boz(p) + tmp
            return acc + baz(2)

        def impl():
            if False:
                i = 10
                return i + 15
            z = 9

            def bar(x):
                if False:
                    print('Hello World!')
                return foo(z) + 7 + x
            return bar(z + 2)
        if utils.PYVERSION in ((3, 8), (3, 9)):
            bc = 33
        elif utils.PYVERSION in ((3, 10), (3, 11)):
            bc = 35
        else:
            raise ValueError(f'Unsupported Python version: {utils.PYVERSION}')
        self.check(impl, inline_expect={'foo': True, 'boz': True, 'fortran': True}, block_count=bc)

    def test_inline_renaming_scheme(self):
        if False:
            i = 10
            return i + 15

        @njit(inline='always')
        def bar(z):
            if False:
                for i in range(10):
                    print('nop')
            x = 5
            y = 10
            return x + y + z

        @njit(pipeline_class=IRPreservingTestPipeline)
        def foo(a, b):
            if False:
                i = 10
                return i + 15
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
            basename = self.id().lstrip(self.__module__)
            regex = f'{basename}__locals__bar_v[0-9]+.x'
            self.assertRegex(name, regex)

class TestRegisterJitableInlining(MemoryLeakMixin, InliningBase):

    def test_register_jitable_inlines(self):
        if False:
            i = 10
            return i + 15

        @register_jitable(inline='always')
        def foo():
            if False:
                print('Hello World!')
            return 1

        def impl():
            if False:
                print('Hello World!')
            foo()
        self.check(impl, inline_expect={'foo': True})

class TestOverloadInlining(MemoryLeakMixin, InliningBase):

    def test_basic_inline_never(self):
        if False:
            print('Hello World!')

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            pass

        @overload(foo, inline='never')
        def foo_overload():
            if False:
                while True:
                    i = 10

            def foo_impl():
                if False:
                    print('Hello World!')
                pass
            return foo_impl

        def impl():
            if False:
                return 10
            return foo()
        self.check(impl, inline_expect={'foo': False})

    def test_basic_inline_always(self):
        if False:
            print('Hello World!')

        def foo():
            if False:
                i = 10
                return i + 15
            pass

        @overload(foo, inline='always')
        def foo_overload():
            if False:
                return 10

            def impl():
                if False:
                    return 10
                pass
            return impl

        def impl():
            if False:
                while True:
                    i = 10
            return foo()
        self.check(impl, inline_expect={'foo': True})

    def test_inline_always_kw_no_default(self):
        if False:
            while True:
                i = 10

        def foo(a, b):
            if False:
                return 10
            return a + b

        @overload(foo, inline='always')
        def overload_foo(a, b):
            if False:
                while True:
                    i = 10
            return lambda a, b: a + b

        def impl():
            if False:
                return 10
            return foo(3, b=4)
        self.check(impl, inline_expect={'foo': True})

    def test_inline_operators_unary(self):
        if False:
            while True:
                i = 10

        def impl_inline(x):
            if False:
                return 10
            return -x

        def impl_noinline(x):
            if False:
                for i in range(10):
                    print('nop')
            return +x
        dummy_unary_impl = lambda x: True
        (Dummy, DummyType) = self.make_dummy_type()
        setattr(Dummy, '__neg__', dummy_unary_impl)
        setattr(Dummy, '__pos__', dummy_unary_impl)

        @overload(operator.neg, inline='always')
        def overload_dummy_neg(x):
            if False:
                i = 10
                return i + 15
            if isinstance(x, DummyType):
                return dummy_unary_impl

        @overload(operator.pos, inline='never')
        def overload_dummy_pos(x):
            if False:
                return 10
            if isinstance(x, DummyType):
                return dummy_unary_impl
        self.check(impl_inline, Dummy(), inline_expect={'neg': True})
        self.check(impl_noinline, Dummy(), inline_expect={'pos': False})

    def test_inline_operators_binop(self):
        if False:
            i = 10
            return i + 15

        def impl_inline(x):
            if False:
                return 10
            return x == 1

        def impl_noinline(x):
            if False:
                for i in range(10):
                    print('nop')
            return x != 1
        (Dummy, DummyType) = self.make_dummy_type()
        dummy_binop_impl = lambda a, b: True
        setattr(Dummy, '__eq__', dummy_binop_impl)
        setattr(Dummy, '__ne__', dummy_binop_impl)

        @overload(operator.eq, inline='always')
        def overload_dummy_eq(a, b):
            if False:
                i = 10
                return i + 15
            if isinstance(a, DummyType):
                return dummy_binop_impl

        @overload(operator.ne, inline='never')
        def overload_dummy_ne(a, b):
            if False:
                i = 10
                return i + 15
            if isinstance(a, DummyType):
                return dummy_binop_impl
        self.check(impl_inline, Dummy(), inline_expect={'eq': True})
        self.check(impl_noinline, Dummy(), inline_expect={'ne': False})

    def test_inline_operators_inplace_binop(self):
        if False:
            i = 10
            return i + 15

        def impl_inline(x):
            if False:
                return 10
            x += 1

        def impl_noinline(x):
            if False:
                while True:
                    i = 10
            x -= 1
        (Dummy, DummyType) = self.make_dummy_type()
        dummy_inplace_binop_impl = lambda a, b: True
        setattr(Dummy, '__iadd__', dummy_inplace_binop_impl)
        setattr(Dummy, '__isub__', dummy_inplace_binop_impl)

        @overload(operator.iadd, inline='always')
        def overload_dummy_iadd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.isub, inline='never')
        def overload_dummy_isub(a, b):
            if False:
                print('Hello World!')
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.add, inline='always')
        def overload_dummy_add(a, b):
            if False:
                print('Hello World!')
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl

        @overload(operator.sub, inline='never')
        def overload_dummy_sub(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(a, DummyType):
                return dummy_inplace_binop_impl
        self.check(impl_inline, Dummy(), inline_expect={'iadd': True})
        self.check(impl_noinline, Dummy(), inline_expect={'isub': False})

    def test_inline_always_operators_getitem(self):
        if False:
            i = 10
            return i + 15

        def impl(x, idx):
            if False:
                return 10
            return x[idx]

        def impl_static_getitem(x):
            if False:
                return 10
            return x[1]
        (Dummy, DummyType) = self.make_dummy_type()
        dummy_getitem_impl = lambda obj, idx: None
        setattr(Dummy, '__getitem__', dummy_getitem_impl)

        @overload(operator.getitem, inline='always')
        def overload_dummy_getitem(obj, idx):
            if False:
                while True:
                    i = 10
            if isinstance(obj, DummyType):
                return dummy_getitem_impl
        self.check(impl, Dummy(), 1, inline_expect={'getitem': True})
        self.check(impl_static_getitem, Dummy(), inline_expect={'getitem': True})

    def test_inline_never_operators_getitem(self):
        if False:
            for i in range(10):
                print('nop')

        def impl(x, idx):
            if False:
                print('Hello World!')
            return x[idx]

        def impl_static_getitem(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[1]
        (Dummy, DummyType) = self.make_dummy_type()
        dummy_getitem_impl = lambda obj, idx: None
        setattr(Dummy, '__getitem__', dummy_getitem_impl)

        @overload(operator.getitem, inline='never')
        def overload_dummy_getitem(obj, idx):
            if False:
                return 10
            if isinstance(obj, DummyType):
                return dummy_getitem_impl
        self.check(impl, Dummy(), 1, inline_expect={'getitem': False})
        self.check(impl_static_getitem, Dummy(), inline_expect={'getitem': False})

    def test_inline_stararg_error(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a, *b):
            if False:
                while True:
                    i = 10
            return a + b[0]

        @overload(foo, inline='always')
        def overload_foo(a, *b):
            if False:
                return 10
            return lambda a, *b: a + b[0]

        def impl():
            if False:
                while True:
                    i = 10
            return foo(3, 3, 5)
        with self.assertRaises(NotImplementedError) as e:
            self.check(impl, inline_expect={'foo': True})
        self.assertIn('Stararg not supported in inliner for arg 1 *b', str(e.exception))

    def test_basic_inline_combos(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                return 10
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for (inline_foo, inline_bar, inline_baz) in product(opts, opts, opts):

            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def bar():
                if False:
                    print('Hello World!')
                pass

            def baz():
                if False:
                    while True:
                        i = 10
                pass

            @overload(foo, inline=inline_foo)
            def foo_overload():
                if False:
                    while True:
                        i = 10

                def impl():
                    if False:
                        return 10
                    return
                return impl

            @overload(bar, inline=inline_bar)
            def bar_overload():
                if False:
                    return 10

                def impl():
                    if False:
                        while True:
                            i = 10
                    return
                return impl

            @overload(baz, inline=inline_baz)
            def baz_overload():
                if False:
                    return 10

                def impl():
                    if False:
                        print('Hello World!')
                    return
                return impl
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_freevar_bindings(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                return 10
            x = foo()
            y = bar()
            z = baz()
            return (x, y, z)
        opts = ('always', 'never')
        for (inline_foo, inline_bar, inline_baz) in product(opts, opts, opts):

            def foo():
                if False:
                    i = 10
                    return i + 15
                x = 10
                y = 20
                z = x + 12
                return (x, y + 3, z)

            def bar():
                if False:
                    while True:
                        i = 10
                x = 30
                y = 40
                z = x + 12
                return (x, y + 3, z)

            def baz():
                if False:
                    return 10
                x = 60
                y = 80
                z = x + 12
                return (x, y + 3, z)

            def factory(target, x, y, inline=None):
                if False:
                    i = 10
                    return i + 15
                z = x + 12

                @overload(target, inline=inline)
                def func():
                    if False:
                        print('Hello World!')

                    def impl():
                        if False:
                            i = 10
                            return i + 15
                        return (x, y + 3, z)
                    return impl
            factory(foo, 10, 20, inline=inline_foo)
            factory(bar, 30, 40, inline=inline_bar)
            factory(baz, 60, 80, inline=inline_baz)
            inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
            self.check(impl, inline_expect=inline_expect)

    def test_global_overload_binding(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                while True:
                    i = 10
            z = 19
            return _global_defn(z)
        self.check(impl, inline_expect={'_global_defn': True})

    def test_inline_from_another_module(self):
        if False:
            print('Hello World!')
        from .inlining_usecases import baz

        def impl():
            if False:
                while True:
                    i = 10
            z = _GLOBAL1 + 2
            return (baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_getattr(self):
        if False:
            while True:
                i = 10
        import numba.tests.inlining_usecases as iuc

        def impl():
            if False:
                print('Hello World!')
            z = _GLOBAL1 + 2
            return (iuc.baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_w_2_getattr(self):
        if False:
            return 10
        import numba.tests.inlining_usecases
        import numba.tests as nt

        def impl():
            if False:
                while True:
                    i = 10
            z = _GLOBAL1 + 2
            return (nt.inlining_usecases.baz(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_from_another_module_as_freevar(self):
        if False:
            i = 10
            return i + 15

        def factory():
            if False:
                while True:
                    i = 10
            from .inlining_usecases import baz

            @njit(inline='always')
            def tmp():
                if False:
                    for i in range(10):
                        print('nop')
                return baz()
            return tmp
        bop = factory()

        def impl():
            if False:
                for i in range(10):
                    print('nop')
            z = _GLOBAL1 + 2
            return (bop(), z)
        self.check(impl, inline_expect={'baz': True})

    def test_inline_w_freevar_from_another_module(self):
        if False:
            for i in range(10):
                print('nop')
        from .inlining_usecases import bop_factory

        def gen(a, b):
            if False:
                for i in range(10):
                    print('nop')
            bar = bop_factory(a)

            def impl():
                if False:
                    return 10
                z = _GLOBAL1 + a * b
                return (bar(), z, a)
            return impl
        impl = gen(10, 20)
        self.check(impl, inline_expect={'bar': True})

    def test_inlining_models(self):
        if False:
            for i in range(10):
                print('nop')

        def s17_caller_model(expr, caller_info, callee_info):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(caller_info.func_ir)

        def s17_callee_model(expr, caller_info, callee_info):
            if False:
                print('Hello World!')
            self.assertIsInstance(expr, ir.Expr)
            self.assertEqual(expr.op, 'call')
            return self.sentinel_17_cost_model(callee_info.func_ir)
        for (caller, callee) in ((10, 11), (17, 11)):

            def foo():
                if False:
                    i = 10
                    return i + 15
                return callee

            @overload(foo, inline=s17_caller_model)
            def foo_ol():
                if False:
                    print('Hello World!')

                def impl():
                    if False:
                        for i in range(10):
                            print('nop')
                    return callee
                return impl

            def impl(z):
                if False:
                    return 10
                x = z + caller
                y = foo()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'foo': caller == 17})
        for (caller, callee) in ((11, 17), (11, 10)):

            def bar():
                if False:
                    i = 10
                    return i + 15
                return callee

            @overload(bar, inline=s17_callee_model)
            def bar_ol():
                if False:
                    i = 10
                    return i + 15

                def impl():
                    if False:
                        for i in range(10):
                            print('nop')
                    return callee
                return impl

            def impl(z):
                if False:
                    i = 10
                    return i + 15
                x = z + caller
                y = bar()
                return (y + 3, x)
            self.check(impl, 10, inline_expect={'bar': callee == 17})

    def test_multiple_overloads_with_different_inline_characteristics(self):
        if False:
            while True:
                i = 10

        def bar(x):
            if False:
                return 10
            if isinstance(typeof(x), types.Float):
                return x + 1234
            else:
                return x + 1

        @overload(bar, inline='always')
        def bar_int_ol(x):
            if False:
                i = 10
                return i + 15
            if isinstance(x, types.Integer):

                def impl(x):
                    if False:
                        print('Hello World!')
                    return x + 1
                return impl

        @overload(bar, inline='never')
        def bar_float_ol(x):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(x, types.Float):

                def impl(x):
                    if False:
                        while True:
                            i = 10
                    return x + 1234
                return impl

        def always_inline_cost_model(*args):
            if False:
                return 10
            return True

        @overload(bar, inline=always_inline_cost_model)
        def bar_complex_ol(x):
            if False:
                print('Hello World!')
            if isinstance(x, types.Complex):

                def impl(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x + 1
                return impl

        def impl():
            if False:
                i = 10
                return i + 15
            a = bar(1)
            b = bar(2.3)
            c = bar(3j)
            return a + b + c
        fir = self.check(impl, inline_expect={'bar': False}, block_count=1)
        block = next(iter(fir.blocks.items()))[1]
        calls = [x for x in block.find_exprs(op='call')]
        self.assertTrue(len(calls) == 1)
        consts = [x.value for x in block.find_insts(ir.Assign) if isinstance(getattr(x, 'value', None), ir.Const)]
        for val in consts:
            self.assertNotEqual(val.value, 1234)

    def test_overload_inline_always_with_literally_in_inlinee(self):
        if False:
            print('Hello World!')

        def foo_ovld(dtype):
            if False:
                while True:
                    i = 10
            if not isinstance(dtype, types.StringLiteral):

                def foo_noop(dtype):
                    if False:
                        i = 10
                        return i + 15
                    return literally(dtype)
                return foo_noop
            if dtype.literal_value == 'str':

                def foo_as_str_impl(dtype):
                    if False:
                        i = 10
                        return i + 15
                    return 10
                return foo_as_str_impl
            if dtype.literal_value in ('int64', 'float64'):

                def foo_as_num_impl(dtype):
                    if False:
                        print('Hello World!')
                    return 20
                return foo_as_num_impl

        def foo(dtype):
            if False:
                print('Hello World!')
            return 10
        overload(foo, inline='always')(foo_ovld)

        def test_impl(dtype):
            if False:
                for i in range(10):
                    print('nop')
            return foo(dtype)
        dtype = 'str'
        self.check(test_impl, dtype, inline_expect={'foo': True})

        def foo(dtype):
            if False:
                return 10
            return 20
        overload(foo, inline='always')(foo_ovld)
        dtype = 'int64'
        self.check(test_impl, dtype, inline_expect={'foo': True})

    def test_inline_always_ssa(self):
        if False:
            while True:
                i = 10
        dummy_true = True

        def foo(A):
            if False:
                for i in range(10):
                    print('nop')
            return True

        @overload(foo, inline='always')
        def foo_overload(A):
            if False:
                for i in range(10):
                    print('nop')

            def impl(A):
                if False:
                    return 10
                s = dummy_true
                for i in range(len(A)):
                    dummy = dummy_true
                    if A[i]:
                        dummy = A[i]
                    s *= dummy
                return s
            return impl

        def impl():
            if False:
                print('Hello World!')
            return foo(np.array([True, False, True]))
        self.check(impl, block_count='SKIP', inline_expect={'foo': True})

    def test_inline_always_ssa_scope_validity(self):
        if False:
            print('Hello World!')

        def bar():
            if False:
                while True:
                    i = 10
            b = 5
            while b > 1:
                b //= 2
            return 10

        @overload(bar, inline='always')
        def bar_impl():
            if False:
                print('Hello World!')
            return bar

        @njit
        def foo():
            if False:
                return 10
            bar()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', errors.NumbaIRAssumptionWarning)
            ignore_internal_warnings()
            self.assertEqual(foo(), foo.py_func())
        self.assertEqual(len(w), 0)

class TestOverloadMethsAttrsInlining(InliningBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_dummy_type()
        super(TestOverloadMethsAttrsInlining, self).setUp()

    def check_method(self, test_impl, args, expected, block_count, expects_inlined=True):
        if False:
            i = 10
            return i + 15
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(j_func(*args), expected)
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            for block in fir.blocks.values():
                calls = list(block.find_exprs('call'))
                self.assertFalse(calls)
        else:
            allcalls = []
            for block in fir.blocks.values():
                allcalls += list(block.find_exprs('call'))
            self.assertTrue(allcalls)

    def check_getattr(self, test_impl, args, expected, block_count, expects_inlined=True):
        if False:
            while True:
                i = 10
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(j_func(*args), expected)
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            for block in fir.blocks.values():
                getattrs = list(block.find_exprs('getattr'))
                self.assertFalse(getattrs)
        else:
            allgetattrs = []
            for block in fir.blocks.values():
                allgetattrs += list(block.find_exprs('getattr'))
            self.assertTrue(allgetattrs)

    def test_overload_method_default_args_always(self):
        if False:
            print('Hello World!')
        (Dummy, DummyType) = self.make_dummy_type()

        @overload_method(DummyType, 'inline_method', inline='always')
        def _get_inlined_method(obj, val=None, val2=None):
            if False:
                while True:
                    i = 10

            def get(obj, val=None, val2=None):
                if False:
                    while True:
                        i = 10
                return ('THIS IS INLINED', val, val2)
            return get

        def foo(obj):
            if False:
                return 10
            return (obj.inline_method(123), obj.inline_method(val2=321))
        self.check_method(test_impl=foo, args=[Dummy()], expected=(('THIS IS INLINED', 123, None), ('THIS IS INLINED', None, 321)), block_count=1)

    def make_overload_method_test(self, costmodel, should_inline):
        if False:
            while True:
                i = 10

        def costmodel(*args):
            if False:
                i = 10
                return i + 15
            return should_inline
        (Dummy, DummyType) = self.make_dummy_type()

        @overload_method(DummyType, 'inline_method', inline=costmodel)
        def _get_inlined_method(obj, val):
            if False:
                return 10

            def get(obj, val):
                if False:
                    return 10
                return ('THIS IS INLINED!!!', val)
            return get

        def foo(obj):
            if False:
                for i in range(10):
                    print('nop')
            return obj.inline_method(123)
        self.check_method(test_impl=foo, args=[Dummy()], expected=('THIS IS INLINED!!!', 123), block_count=1, expects_inlined=should_inline)

    def test_overload_method_cost_driven_always(self):
        if False:
            print('Hello World!')
        self.make_overload_method_test(costmodel='always', should_inline=True)

    def test_overload_method_cost_driven_never(self):
        if False:
            return 10
        self.make_overload_method_test(costmodel='never', should_inline=False)

    def test_overload_method_cost_driven_must_inline(self):
        if False:
            print('Hello World!')
        self.make_overload_method_test(costmodel=lambda *args: True, should_inline=True)

    def test_overload_method_cost_driven_no_inline(self):
        if False:
            return 10
        self.make_overload_method_test(costmodel=lambda *args: False, should_inline=False)

    def make_overload_attribute_test(self, costmodel, should_inline):
        if False:
            i = 10
            return i + 15
        (Dummy, DummyType) = self.make_dummy_type()

        @overload_attribute(DummyType, 'inlineme', inline=costmodel)
        def _get_inlineme(obj):
            if False:
                print('Hello World!')

            def get(obj):
                if False:
                    for i in range(10):
                        print('nop')
                return 'MY INLINED ATTRS'
            return get

        def foo(obj):
            if False:
                i = 10
                return i + 15
            return obj.inlineme
        self.check_getattr(test_impl=foo, args=[Dummy()], expected='MY INLINED ATTRS', block_count=1, expects_inlined=should_inline)

    def test_overload_attribute_always(self):
        if False:
            i = 10
            return i + 15
        self.make_overload_attribute_test(costmodel='always', should_inline=True)

    def test_overload_attribute_never(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_overload_attribute_test(costmodel='never', should_inline=False)

    def test_overload_attribute_costmodel_must_inline(self):
        if False:
            print('Hello World!')
        self.make_overload_attribute_test(costmodel=lambda *args: True, should_inline=True)

    def test_overload_attribute_costmodel_no_inline(self):
        if False:
            while True:
                i = 10
        self.make_overload_attribute_test(costmodel=lambda *args: False, should_inline=False)

class TestGeneralInlining(MemoryLeakMixin, InliningBase):

    def test_with_inlined_and_noninlined_variants(self):
        if False:
            while True:
                i = 10

        @overload(len, inline='always')
        def overload_len(A):
            if False:
                while True:
                    i = 10
            if False:
                return lambda A: 10

        def impl():
            if False:
                print('Hello World!')
            return len([2, 3, 4])
        self.check(impl, inline_expect={'len': False})

    def test_with_kwargs(self):
        if False:
            while True:
                i = 10

        def foo(a, b=3, c=5):
            if False:
                return 10
            return a + b + c

        @overload(foo, inline='always')
        def overload_foo(a, b=3, c=5):
            if False:
                i = 10
                return i + 15

            def impl(a, b=3, c=5):
                if False:
                    while True:
                        i = 10
                return a + b + c
            return impl

        def impl():
            if False:
                print('Hello World!')
            return foo(3, c=10)
        self.check(impl, inline_expect={'foo': True})

    def test_with_kwargs2(self):
        if False:
            while True:
                i = 10

        @njit(inline='always')
        def bar(a, b=12, c=9):
            if False:
                print('Hello World!')
            return a + b

        def impl(a, b=7, c=5):
            if False:
                while True:
                    i = 10
            return bar(a + b, c=19)
        self.check(impl, 3, 4, inline_expect={'bar': True})

    def test_inlining_optional_constant(self):
        if False:
            while True:
                i = 10

        @njit(inline='always')
        def bar(a=None, b=None):
            if False:
                return 10
            if b is None:
                b = 123
            return (a, b)

        def impl():
            if False:
                i = 10
                return i + 15
            return (bar(), bar(123), bar(b=321))
        self.check(impl, block_count='SKIP', inline_expect={'bar': True})

class TestInlineOptions(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        always = InlineOptions('always')
        self.assertTrue(always.is_always_inline)
        self.assertFalse(always.is_never_inline)
        self.assertFalse(always.has_cost_model)
        self.assertEqual(always.value, 'always')
        never = InlineOptions('never')
        self.assertFalse(never.is_always_inline)
        self.assertTrue(never.is_never_inline)
        self.assertFalse(never.has_cost_model)
        self.assertEqual(never.value, 'never')

        def cost_model(x):
            if False:
                return 10
            return x
        model = InlineOptions(cost_model)
        self.assertFalse(model.is_always_inline)
        self.assertFalse(model.is_never_inline)
        self.assertTrue(model.has_cost_model)
        self.assertIs(model.value, cost_model)

class TestInlineMiscIssues(TestCase):

    def test_issue4691(self):
        if False:
            for i in range(10):
                print('nop')

        def output_factory(array, dtype):
            if False:
                while True:
                    i = 10
            pass

        @overload(output_factory, inline='always')
        def ol_output_factory(array, dtype):
            if False:
                return 10
            if isinstance(array, types.npytypes.Array):

                def impl(array, dtype):
                    if False:
                        print('Hello World!')
                    shape = array.shape[3:]
                    return np.zeros(shape, dtype=dtype)
                return impl

        @njit(nogil=True)
        def fn(array):
            if False:
                print('Hello World!')
            out = output_factory(array, array.dtype)
            return out

        @njit(nogil=True)
        def fn2(array):
            if False:
                return 10
            return np.zeros(array.shape[3:], dtype=array.dtype)
        fn(np.ones((10, 20, 30, 40, 50)))
        fn2(np.ones((10, 20, 30, 40, 50)))

    def test_issue4693(self):
        if False:
            for i in range(10):
                print('nop')

        @njit(inline='always')
        def inlining(array):
            if False:
                return 10
            if array.ndim != 1:
                raise ValueError('Invalid number of dimensions')
            return array

        @njit
        def fn(array):
            if False:
                while True:
                    i = 10
            return inlining(array)
        fn(np.zeros(10))

    def test_issue5476(self):
        if False:
            i = 10
            return i + 15

        @njit(inline='always')
        def inlining():
            if False:
                return 10
            msg = 'Something happened'
            raise ValueError(msg)

        @njit
        def fn():
            if False:
                return 10
            return inlining()
        with self.assertRaises(ValueError) as raises:
            fn()
        self.assertIn('Something happened', str(raises.exception))

    def test_issue5792(self):
        if False:
            i = 10
            return i + 15

        class Dummy:

            def __init__(self, data):
                if False:
                    print('Hello World!')
                self.data = data

            def div(self, other):
                if False:
                    print('Hello World!')
                return data / other.data

        class DummyType(types.Type):

            def __init__(self, data):
                if False:
                    return 10
                self.data = data
                super().__init__(name=f'Dummy({self.data})')

        @register_model(DummyType)
        class DummyTypeModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                if False:
                    for i in range(10):
                        print('nop')
                members = [('data', fe_type.data)]
                super().__init__(dmm, fe_type, members)
        make_attribute_wrapper(DummyType, 'data', '_data')

        @intrinsic
        def init_dummy(typingctx, data):
            if False:
                i = 10
                return i + 15

            def codegen(context, builder, sig, args):
                if False:
                    print('Hello World!')
                typ = sig.return_type
                (data,) = args
                dummy = cgutils.create_struct_proxy(typ)(context, builder)
                dummy.data = data
                if context.enable_nrt:
                    context.nrt.incref(builder, sig.args[0], data)
                return dummy._getvalue()
            ret_typ = DummyType(data)
            sig = signature(ret_typ, data)
            return (sig, codegen)

        @overload(Dummy, inline='always')
        def dummy_overload(data):
            if False:
                for i in range(10):
                    print('nop')

            def ctor(data):
                if False:
                    print('Hello World!')
                return init_dummy(data)
            return ctor

        @overload_method(DummyType, 'div', inline='always')
        def div_overload(self, other):
            if False:
                while True:
                    i = 10

            def impl(self, other):
                if False:
                    return 10
                return self._data / other._data
            return impl

        @njit
        def test_impl(data, other_data):
            if False:
                i = 10
                return i + 15
            dummy = Dummy(data)
            other = Dummy(other_data)
            return dummy.div(other)
        data = 1.0
        other_data = 2.0
        res = test_impl(data, other_data)
        self.assertEqual(res, data / other_data)

    def test_issue5824(self):
        if False:
            return 10
        ' Similar to the above test_issue5792, checks mutation of the inlinee\n        IR is local only'

        class CustomCompiler(CompilerBase):

            def define_pipelines(self):
                if False:
                    while True:
                        i = 10
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(InlineOverloads, InlineOverloads)
                pm.finalize()
                return [pm]

        def bar(x):
            if False:
                while True:
                    i = 10
            ...

        @overload(bar, inline='always')
        def ol_bar(x):
            if False:
                while True:
                    i = 10
            if isinstance(x, types.Integer):

                def impl(x):
                    if False:
                        while True:
                            i = 10
                    return x + 1.3
                return impl

        @njit(pipeline_class=CustomCompiler)
        def foo(z):
            if False:
                while True:
                    i = 10
            return (bar(z), bar(z))
        self.assertEqual(foo(10), (11.3, 11.3))

    @skip_parfors_unsupported
    def test_issue7380(self):
        if False:
            for i in range(10):
                print('nop')

        @njit(inline='always')
        def bar(x):
            if False:
                i = 10
                return i + 15
            for i in range(x.size):
                x[i] += 1

        @njit(parallel=True)
        def foo(a):
            if False:
                while True:
                    i = 10
            for i in prange(a.shape[0]):
                bar(a[i])
        a = np.ones((10, 10))
        foo(a)
        self.assertPreciseEqual(a, 2 * np.ones_like(a))

        @njit(parallel=True)
        def foo_bad(a):
            if False:
                return 10
            for i in prange(a.shape[0]):
                x = a[i]
                for i in range(x.size):
                    x[i] += 1
        with self.assertRaises(errors.UnsupportedRewriteError) as e:
            foo_bad(a)
        self.assertIn('Overwrite of parallel loop index', str(e.exception))
if __name__ == '__main__':
    unittest.main()