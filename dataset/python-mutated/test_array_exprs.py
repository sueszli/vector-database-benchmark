import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import MemoryLeakMixin, TestCase, temp_directory, create_temp_module
from numba.extending import overload, models, lower_builtin, register_model, make_attribute_wrapper, type_callable, typeof_impl
import operator
import textwrap
import unittest

class Namespace(dict):

    def __getattr__(s, k):
        if False:
            while True:
                i = 10
        return s[k] if k in s else super(Namespace, s).__getattr__(k)

def axy(a, x, y):
    if False:
        i = 10
        return i + 15
    return a * x + y

def ax2(a, x, y):
    if False:
        for i in range(10):
            print('nop')
    return a * x + y

def pos_root(As, Bs, Cs):
    if False:
        for i in range(10):
            print('nop')
    return (-Bs + (Bs ** 2.0 - 4.0 * As * Cs) ** 0.5) / (2.0 * As)

def neg_root_common_subexpr(As, Bs, Cs):
    if False:
        i = 10
        return i + 15
    _2As = 2.0 * As
    _4AsCs = 2.0 * _2As * Cs
    _Bs2_4AsCs = Bs ** 2.0 - _4AsCs
    return (-Bs - _Bs2_4AsCs ** 0.5) / _2As

def neg_root_complex_subexpr(As, Bs, Cs):
    if False:
        i = 10
        return i + 15
    _2As = 2.0 * As
    _4AsCs = 2.0 * _2As * Cs
    _Bs2_4AsCs = Bs ** 2.0 - _4AsCs + 0j
    return (-Bs - _Bs2_4AsCs ** 0.5) / _2As
vaxy = vectorize(axy)

def call_stuff(a0, a1):
    if False:
        print('Hello World!')
    return np.cos(vaxy(a0, np.sin(a1) - 1.0, 1.0))

def are_roots_imaginary(As, Bs, Cs):
    if False:
        for i in range(10):
            print('nop')
    return Bs ** 2 - 4 * As * Cs < 0

def div_add(As, Bs, Cs):
    if False:
        while True:
            i = 10
    return As / Bs + Cs

def cube(As):
    if False:
        while True:
            i = 10
    return As ** 3

def explicit_output(a, b, out):
    if False:
        return 10
    np.cos(a, out)
    return np.add(out, b, out)

def variable_name_reuse(a, b, c, d):
    if False:
        for i in range(10):
            print('nop')
    u = a + b
    u = u - a * b
    u = u * c + d
    return u

def distance_matrix(vectors):
    if False:
        while True:
            i = 10
    n_vectors = vectors.shape[0]
    result = np.empty((n_vectors, n_vectors), dtype=np.float64)
    for i in range(n_vectors):
        for j in range(i, n_vectors):
            result[i, j] = result[j, i] = np.sum((vectors[i] - vectors[j]) ** 2) ** 0.5
    return result

class RewritesTester(Compiler):

    @classmethod
    def mk_pipeline(cls, args, return_type=None, flags=None, locals={}, library=None, typing_context=None, target_context=None):
        if False:
            print('Hello World!')
        if not flags:
            flags = Flags()
        flags.nrt = True
        if typing_context is None:
            typing_context = cpu_target.typing_context
        if target_context is None:
            target_context = cpu_target.target_context
        return cls(typing_context, target_context, library, args, return_type, flags, locals)

    @classmethod
    def mk_no_rw_pipeline(cls, args, return_type=None, flags=None, locals={}, library=None, **kws):
        if False:
            i = 10
            return i + 15
        if not flags:
            flags = Flags()
        flags.no_rewrites = True
        return cls.mk_pipeline(args, return_type, flags, locals, library, **kws)

class TestArrayExpressions(MemoryLeakMixin, TestCase):

    def _compile_function(self, fn, arg_tys):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compile the given function both without and with rewrites enabled.\n        '
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_0 = control_pipeline.compile_extra(fn)
        control_cfunc = cres_0.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        cres_1 = test_pipeline.compile_extra(fn)
        test_cfunc = cres_1.entry_point
        return (control_pipeline, control_cfunc, test_pipeline, test_cfunc)

    def test_simple_expr(self):
        if False:
            print('Hello World!')
        '\n        Using a simple array expression, verify that rewriting is taking\n        place, and is fusing loops.\n        '
        A = np.linspace(0, 1, 10)
        X = np.linspace(2, 1, 10)
        Y = np.linspace(1, 2, 10)
        arg_tys = [typeof(arg) for arg in (A, X, Y)]
        (control_pipeline, nb_axy_0, test_pipeline, nb_axy_1) = self._compile_function(axy, arg_tys)
        control_pipeline2 = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_2 = control_pipeline2.compile_extra(ax2)
        nb_ctl = cres_2.entry_point
        expected = nb_axy_0(A, X, Y)
        actual = nb_axy_1(A, X, Y)
        control = nb_ctl(A, X, Y)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(control, actual)
        ir0 = control_pipeline.state.func_ir.blocks
        ir1 = test_pipeline.state.func_ir.blocks
        ir2 = control_pipeline2.state.func_ir.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertEqual(len(ir0), len(ir2))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(ir0[0].body), len(ir2[0].body))

    def _get_array_exprs(self, block):
        if False:
            print('Hello World!')
        for instr in block:
            if isinstance(instr, ir.Assign):
                if isinstance(instr.value, ir.Expr):
                    if instr.value.op == 'arrayexpr':
                        yield instr

    def _array_expr_to_set(self, expr, out=None):
        if False:
            while True:
                i = 10
        '\n        Convert an array expression tree into a set of operators.\n        '
        if out is None:
            out = set()
        if not isinstance(expr, tuple):
            raise ValueError('{0} not a tuple'.format(expr))
        (operation, operands) = expr
        processed_operands = []
        for operand in operands:
            if isinstance(operand, tuple):
                (operand, _) = self._array_expr_to_set(operand, out)
            processed_operands.append(operand)
        processed_expr = (operation, tuple(processed_operands))
        out.add(processed_expr)
        return (processed_expr, out)

    def _test_root_function(self, fn=pos_root):
        if False:
            while True:
                i = 10
        A = np.random.random(10)
        B = np.random.random(10) + 1.0
        C = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B, C)]
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        control_cres = control_pipeline.compile_extra(fn)
        nb_fn_0 = control_cres.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        test_cres = test_pipeline.compile_extra(fn)
        nb_fn_1 = test_cres.entry_point
        np_result = fn(A, B, C)
        nb_result_0 = nb_fn_0(A, B, C)
        nb_result_1 = nb_fn_1(A, B, C)
        np.testing.assert_array_almost_equal(np_result, nb_result_0)
        np.testing.assert_array_almost_equal(nb_result_0, nb_result_1)
        return Namespace(locals())

    def _test_cube_function(self, fn=cube):
        if False:
            i = 10
            return i + 15
        A = np.arange(10, dtype=np.float64)
        arg_tys = (typeof(A),)
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        control_cres = control_pipeline.compile_extra(fn)
        nb_fn_0 = control_cres.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        test_cres = test_pipeline.compile_extra(fn)
        nb_fn_1 = test_cres.entry_point
        expected = A ** 3
        self.assertPreciseEqual(expected, nb_fn_0(A))
        self.assertPreciseEqual(expected, nb_fn_1(A))
        return Namespace(locals())

    def _test_explicit_output_function(self, fn):
        if False:
            i = 10
            return i + 15
        '\n        Test function having a (a, b, out) signature where *out* is\n        an output array the function writes into.\n        '
        A = np.arange(10, dtype=np.float64)
        B = A + 1
        arg_tys = (typeof(A),) * 3
        (control_pipeline, control_cfunc, test_pipeline, test_cfunc) = self._compile_function(fn, arg_tys)

        def run_func(fn):
            if False:
                for i in range(10):
                    print('nop')
            out = np.zeros_like(A)
            fn(A, B, out)
            return out
        expected = run_func(fn)
        self.assertPreciseEqual(expected, run_func(control_cfunc))
        self.assertPreciseEqual(expected, run_func(test_cfunc))
        return Namespace(locals())

    def _assert_array_exprs(self, block, expected_count):
        if False:
            return 10
        '\n        Assert the *block* has the expected number of array expressions\n        in it.\n        '
        rewrite_count = len(list(self._get_array_exprs(block)))
        self.assertEqual(rewrite_count, expected_count)

    def _assert_total_rewrite(self, control_ir, test_ir, trivial=False):
        if False:
            print('Hello World!')
        '\n        Given two dictionaries of Numba IR blocks, check to make sure the\n        control IR has no array expressions, while the test IR\n        contains one and only one.\n        '
        self.assertEqual(len(control_ir), len(test_ir))
        control_block = control_ir[0].body
        test_block = test_ir[0].body
        self._assert_array_exprs(control_block, 0)
        self._assert_array_exprs(test_block, 1)
        if not trivial:
            self.assertGreater(len(control_block), len(test_block))

    def _assert_no_rewrite(self, control_ir, test_ir):
        if False:
            print('Hello World!')
        '\n        Given two dictionaries of Numba IR blocks, check to make sure\n        the control IR and the test IR both have no array expressions.\n        '
        self.assertEqual(len(control_ir), len(test_ir))
        for (k, v) in control_ir.items():
            control_block = v.body
            test_block = test_ir[k].body
            self.assertEqual(len(control_block), len(test_block))
            self._assert_array_exprs(control_block, 0)
            self._assert_array_exprs(test_block, 0)

    def test_trivial_expr(self):
        if False:
            while True:
                i = 10
        '\n        Ensure even a non-nested expression is rewritten, as it can enable\n        scalar optimizations such as rewriting `x ** 2`.\n        '
        ns = self._test_cube_function()
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks, trivial=True)

    def test_complicated_expr(self):
        if False:
            print('Hello World!')
        '\n        Using the polynomial root function, ensure the full expression is\n        being put in the same kernel with no remnants of intermediate\n        array expressions.\n        '
        ns = self._test_root_function()
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)

    def test_common_subexpressions(self, fn=neg_root_common_subexpr):
        if False:
            return 10
        '\n        Attempt to verify that rewriting will incorporate user common\n        subexpressions properly.\n        '
        ns = self._test_root_function(fn)
        ir0 = ns.control_pipeline.state.func_ir.blocks
        ir1 = ns.test_pipeline.state.func_ir.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
        array_expr_instrs = list(self._get_array_exprs(ir1[0].body))
        self.assertGreater(len(array_expr_instrs), 1)
        array_sets = list((self._array_expr_to_set(instr.value.expr)[1] for instr in array_expr_instrs))
        for (expr_set_0, expr_set_1) in zip(array_sets[:-1], array_sets[1:]):
            intersections = expr_set_0.intersection(expr_set_1)
            if intersections:
                self.fail('Common subexpressions detected in array expressions ({0})'.format(intersections))

    def test_complex_subexpression(self):
        if False:
            i = 10
            return i + 15
        return self.test_common_subexpressions(neg_root_complex_subexpr)

    def test_ufunc_and_dufunc_calls(self):
        if False:
            return 10
        '\n        Verify that ufunc and DUFunc calls are being properly included in\n        array expressions.\n        '
        A = np.random.random(10)
        B = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B)]
        vaxy_descr = vaxy._dispatcher.targetdescr
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
        cres_0 = control_pipeline.compile_extra(call_stuff)
        nb_call_stuff_0 = cres_0.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
        cres_1 = test_pipeline.compile_extra(call_stuff)
        nb_call_stuff_1 = cres_1.entry_point
        expected = call_stuff(A, B)
        control = nb_call_stuff_0(A, B)
        actual = nb_call_stuff_1(A, B)
        np.testing.assert_array_almost_equal(expected, control)
        np.testing.assert_array_almost_equal(expected, actual)
        self._assert_total_rewrite(control_pipeline.state.func_ir.blocks, test_pipeline.state.func_ir.blocks)

    def test_cmp_op(self):
        if False:
            return 10
        '\n        Verify that comparison operators are supported by the rewriter.\n        '
        ns = self._test_root_function(are_roots_imaginary)
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)

    def test_explicit_output(self):
        if False:
            while True:
                i = 10
        '\n        Check that ufunc calls with explicit outputs are not rewritten.\n        '
        ns = self._test_explicit_output_function(explicit_output)
        self._assert_no_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)

class TestRewriteIssues(MemoryLeakMixin, TestCase):

    def test_issue_1184(self):
        if False:
            print('Hello World!')
        from numba import jit
        import numpy as np

        @jit(nopython=True)
        def foo(arr):
            if False:
                return 10
            return arr

        @jit(nopython=True)
        def bar(arr):
            if False:
                return 10
            c = foo(arr)
            d = foo(arr)
            return (c, d)
        arr = np.arange(10)
        (out_c, out_d) = bar(arr)
        self.assertIs(out_c, out_d)
        self.assertIs(out_c, arr)

    def test_issue_1264(self):
        if False:
            print('Hello World!')
        n = 100
        x = np.random.uniform(size=n * 3).reshape((n, 3))
        expected = distance_matrix(x)
        actual = njit(distance_matrix)(x)
        np.testing.assert_array_almost_equal(expected, actual)
        gc.collect()

    def test_issue_1372(self):
        if False:
            for i in range(10):
                print('nop')
        'Test array expression with duplicated term'
        from numba import njit

        @njit
        def foo(a, b):
            if False:
                while True:
                    i = 10
            b = np.sin(b)
            return b + b + a
        a = np.random.uniform(10)
        b = np.random.uniform(10)
        expect = foo.py_func(a, b)
        got = foo(a, b)
        np.testing.assert_allclose(got, expect)

    def test_unary_arrayexpr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Typing of unary array expression (np.negate) can be incorrect.\n        '

        @njit
        def foo(a, b):
            if False:
                print('Hello World!')
            return b - a + -a
        b = 1.5
        a = np.arange(10, dtype=np.int32)
        expect = foo.py_func(a, b)
        got = foo(a, b)
        self.assertPreciseEqual(got, expect)

    def test_bitwise_arrayexpr(self):
        if False:
            while True:
                i = 10
        '\n        Typing of bitwise boolean array expression can be incorrect\n        (issue #1813).\n        '

        @njit
        def foo(a, b):
            if False:
                while True:
                    i = 10
            return ~(a & ~b)
        a = np.array([True, True, False, False])
        b = np.array([False, True, False, True])
        expect = foo.py_func(a, b)
        got = foo(a, b)
        self.assertPreciseEqual(got, expect)

    def test_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Type annotation of array expressions with disambiguated\n        variable names (issue #1466).\n        '
        cfunc = njit(variable_name_reuse)
        a = np.linspace(0, 1, 10)
        cfunc(a, a, a, a)
        buf = StringIO()
        cfunc.inspect_types(buf)
        res = buf.getvalue()
        self.assertIn('#   u.1 = ', res)
        self.assertIn('#   u.2 = ', res)

    def test_issue_5599_name_collision(self):
        if False:
            print('Hello World!')

        @njit
        def f(x):
            if False:
                while True:
                    i = 10
            arr = np.ones(x)
            for _ in range(2):
                val = arr * arr
                arr = arr.copy()
            return arr
        got = f(5)
        expect = f.py_func(5)
        np.testing.assert_array_equal(got, expect)

class TestSemantics(MemoryLeakMixin, unittest.TestCase):

    def test_division_by_zero(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = div_add
        cfunc = njit(pyfunc)
        a = np.float64([0.0, 1.0, float('inf')])
        b = np.float64([0.0, 0.0, 1.0])
        c = np.ones_like(a)
        expect = pyfunc(a, b, c)
        got = cfunc(a, b, c)
        np.testing.assert_array_equal(expect, got)

class TestOptionals(MemoryLeakMixin, unittest.TestCase):
    """ Tests the arrival and correct lowering of Optional types at a arrayexpr
    derived ufunc, see #3972"""

    def test_optional_scalar_type(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def arr_expr(x, y):
            if False:
                return 10
            return x + y

        @njit
        def do_call(x, y):
            if False:
                print('Hello World!')
            if y > 0:
                z = None
            else:
                z = y
            return arr_expr(x, z)
        args = (np.arange(5), -1.2)
        res = do_call(*args)
        expected = do_call.py_func(*args)
        np.testing.assert_allclose(res, expected)
        s = arr_expr.signatures
        oty = s[0][1]
        self.assertTrue(isinstance(oty, types.Optional))
        self.assertTrue(isinstance(oty.type, types.Float))

    def test_optional_array_type(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def arr_expr(x, y):
            if False:
                print('Hello World!')
            return x + y

        @njit
        def do_call(x, y):
            if False:
                for i in range(10):
                    print('nop')
            if y[0] > 0:
                z = None
            else:
                z = y
            return arr_expr(x, z)
        args = (np.arange(5), np.arange(5.0))
        res = do_call(*args)
        expected = do_call.py_func(*args)
        np.testing.assert_allclose(res, expected)
        s = arr_expr.signatures
        oty = s[0][1]
        self.assertTrue(isinstance(oty, types.Optional))
        self.assertTrue(isinstance(oty.type, types.Array))
        self.assertTrue(isinstance(oty.type.dtype, types.Float))

class TestOptionalsExceptions(MemoryLeakMixin, unittest.TestCase):

    def test_optional_scalar_type_exception_on_none(self):
        if False:
            while True:
                i = 10
        self.disable_leak_check()

        @njit
        def arr_expr(x, y):
            if False:
                return 10
            return x + y

        @njit
        def do_call(x, y):
            if False:
                i = 10
                return i + 15
            if y > 0:
                z = None
            else:
                z = y
            return arr_expr(x, z)
        args = (np.arange(5), 1.0)
        with self.assertRaises(TypeError) as raises:
            do_call(*args)
        self.assertIn('expected float64, got None', str(raises.exception))
        s = arr_expr.signatures
        oty = s[0][1]
        self.assertTrue(isinstance(oty, types.Optional))
        self.assertTrue(isinstance(oty.type, types.Float))

    def test_optional_array_type_exception_on_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.disable_leak_check()

        @njit
        def arr_expr(x, y):
            if False:
                return 10
            return x + y

        @njit
        def do_call(x, y):
            if False:
                print('Hello World!')
            if y[0] > 0:
                z = None
            else:
                z = y
            return arr_expr(x, z)
        args = (np.arange(5), np.arange(1.0, 5.0))
        with self.assertRaises(TypeError) as raises:
            do_call(*args)
        excstr = str(raises.exception)
        self.assertIn('expected array(float64,', excstr)
        self.assertIn('got None', excstr)
        s = arr_expr.signatures
        oty = s[0][1]
        self.assertTrue(isinstance(oty, types.Optional))
        self.assertTrue(isinstance(oty.type, types.Array))
        self.assertTrue(isinstance(oty.type.dtype, types.Float))

class TestExternalTypes(MemoryLeakMixin, unittest.TestCase):
    """ Tests RewriteArrayExprs with external (user defined) types,
    see #5157"""
    source_lines = textwrap.dedent("\n        from numba.core import types\n\n        class FooType(types.Type):\n            def __init__(self):\n                super(FooType, self).__init__(name='Foo')\n        ")

    def make_foo_type(self, FooType):
        if False:
            for i in range(10):
                print('nop')

        class Foo(object):

            def __init__(self, value):
                if False:
                    while True:
                        i = 10
                self.value = value

        @register_model(FooType)
        class FooModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                if False:
                    print('Hello World!')
                members = [('value', types.intp)]
                models.StructModel.__init__(self, dmm, fe_type, members)
        make_attribute_wrapper(FooType, 'value', 'value')

        @type_callable(Foo)
        def type_foo(context):
            if False:
                i = 10
                return i + 15

            def typer(value):
                if False:
                    print('Hello World!')
                return FooType()
            return typer

        @lower_builtin(Foo, types.intp)
        def impl_foo(context, builder, sig, args):
            if False:
                while True:
                    i = 10
            typ = sig.return_type
            [value] = args
            foo = cgutils.create_struct_proxy(typ)(context, builder)
            foo.value = value
            return foo._getvalue()

        @typeof_impl.register(Foo)
        def typeof_foo(val, c):
            if False:
                return 10
            return FooType()
        return (Foo, FooType)

    def test_external_type(self):
        if False:
            for i in range(10):
                print('nop')
        with create_temp_module(self.source_lines) as test_module:
            (Foo, FooType) = self.make_foo_type(test_module.FooType)

            @overload(operator.add)
            def overload_foo_add(lhs, rhs):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(lhs, FooType) and isinstance(rhs, types.Array):

                    def imp(lhs, rhs):
                        if False:
                            print('Hello World!')
                        return np.array([lhs.value, rhs[0]])
                    return imp

            @overload(operator.add)
            def overload_foo_add(lhs, rhs):
                if False:
                    return 10
                if isinstance(lhs, FooType) and isinstance(rhs, FooType):

                    def imp(lhs, rhs):
                        if False:
                            return 10
                        return np.array([lhs.value, rhs.value])
                    return imp

            @overload(operator.neg)
            def overload_foo_neg(x):
                if False:
                    while True:
                        i = 10
                if isinstance(x, FooType):

                    def imp(x):
                        if False:
                            print('Hello World!')
                        return np.array([-x.value])
                    return imp

            @njit
            def arr_expr_sum1(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return Foo(x) + np.array([y])

            @njit
            def arr_expr_sum2(x, y):
                if False:
                    i = 10
                    return i + 15
                return Foo(x) + Foo(y)

            @njit
            def arr_expr_neg(x):
                if False:
                    for i in range(10):
                        print('nop')
                return -Foo(x)
            np.testing.assert_array_equal(arr_expr_sum1(0, 1), np.array([0, 1]))
            np.testing.assert_array_equal(arr_expr_sum2(2, 3), np.array([2, 3]))
            np.testing.assert_array_equal(arr_expr_neg(4), np.array([-4]))
if __name__ == '__main__':
    unittest.main()