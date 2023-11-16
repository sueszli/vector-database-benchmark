"""
Tests for SSA reconstruction
"""
import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
_DEBUG = False
if _DEBUG:
    ssa_logger = logging.getLogger('numba.core.ssa')
    ssa_logger.setLevel(level=logging.DEBUG)
    ssa_logger.addHandler(logging.StreamHandler(sys.stderr))

class SSABaseTest(TestCase):

    def check_func(self, func, *args):
        if False:
            i = 10
            return i + 15
        got = func(*copy.deepcopy(args))
        exp = func.py_func(*copy.deepcopy(args))
        self.assertEqual(got, exp)

class TestSSA(SSABaseTest):
    """
    Contains tests to help isolate problems in SSA
    """

    def test_argument_name_reused(self):
        if False:
            return 10

        @njit
        def foo(x):
            if False:
                i = 10
                return i + 15
            x += 1
            return x
        self.check_func(foo, 123)

    def test_if_else_redefine(self):
        if False:
            print('Hello World!')

        @njit
        def foo(x, y):
            if False:
                return 10
            z = x * y
            if x < y:
                z = x
            else:
                z = y
            return z
        self.check_func(foo, 3, 2)
        self.check_func(foo, 2, 3)

    def test_sum_loop(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(n):
            if False:
                for i in range(10):
                    print('nop')
            c = 0
            for i in range(n):
                c += i
            return c
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_loop_2vars(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(n):
            if False:
                while True:
                    i = 10
            c = 0
            d = n
            for i in range(n):
                c += i
                d += n
            return (c, d)
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def test_sum_2d_loop(self):
        if False:
            print('Hello World!')

        @njit
        def foo(n):
            if False:
                return 10
            c = 0
            for i in range(n):
                for j in range(n):
                    c += j
                c += i
            return c
        self.check_func(foo, 0)
        self.check_func(foo, 10)

    def check_undefined_var(self, should_warn):
        if False:
            while True:
                i = 10

        @njit
        def foo(n):
            if False:
                print('Hello World!')
            if n:
                if n > 0:
                    c = 0
                return c
            else:
                c += 1
                return c
        if should_warn:
            with self.assertWarns(errors.NumbaWarning) as warns:
                self.check_func(foo, 1)
            self.assertIn('Detected uninitialized variable c', str(warns.warning))
        else:
            self.check_func(foo, 1)
        with self.assertRaises(UnboundLocalError):
            foo.py_func(0)

    def test_undefined_var(self):
        if False:
            while True:
                i = 10
        with override_config('ALWAYS_WARN_UNINIT_VAR', 0):
            self.check_undefined_var(should_warn=False)
        with override_config('ALWAYS_WARN_UNINIT_VAR', 1):
            self.check_undefined_var(should_warn=True)

    def test_phi_propagation(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(actions):
            if False:
                i = 10
                return i + 15
            n = 1
            i = 0
            ct = 0
            while n > 0 and i < len(actions):
                n -= 1
                while actions[i]:
                    if actions[i]:
                        if actions[i]:
                            n += 10
                        actions[i] -= 1
                    else:
                        if actions[i]:
                            n += 20
                        actions[i] += 1
                    ct += n
                ct += n
            return (ct, n)
        self.check_func(foo, np.array([1, 2]))

    def test_unhandled_undefined(self):
        if False:
            print('Hello World!')

        def function1(arg1, arg2, arg3, arg4, arg5):
            if False:
                while True:
                    i = 10
            if arg1:
                var1 = arg2
                var2 = arg3
                var3 = var2
                var4 = arg1
                return
            else:
                if arg2:
                    if arg4:
                        var5 = arg4
                        return
                    else:
                        var6 = var4
                        return
                    return var6
                else:
                    if arg5:
                        if var1:
                            if arg5:
                                var1 = var6
                                return
                            else:
                                var7 = arg2
                                return arg2
                            return
                        else:
                            if var2:
                                arg5 = arg2
                                return arg1
                            else:
                                var6 = var3
                                return var4
                            return
                        return
                    else:
                        var8 = var1
                        return
                    return var8
                var9 = var3
                var10 = arg5
                return var1
        expect = function1(2, 3, 6, 0, 7)
        got = njit(function1)(2, 3, 6, 0, 7)
        self.assertEqual(expect, got)

class TestReportedSSAIssues(SSABaseTest):

    def test_issue2194(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            V = np.empty(1)
            s = np.uint32(1)
            for i in range(s):
                V[i] = 1
            for i in range(s, 1):
                pass
        self.check_func(foo)

    def test_issue3094(self):
        if False:
            print('Hello World!')

        @njit
        def doit(x):
            if False:
                i = 10
                return i + 15
            return x

        @njit
        def foo(pred):
            if False:
                i = 10
                return i + 15
            if pred:
                x = True
            else:
                x = False
            return doit(x)
        self.check_func(foo, False)

    def test_issue3931(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(arr):
            if False:
                return 10
            for i in range(1):
                arr = arr.reshape(3 * 2)
                arr = arr.reshape(3, 2)
            return arr
        np.testing.assert_allclose(foo(np.zeros((3, 2))), foo.py_func(np.zeros((3, 2))))

    def test_issue3976(self):
        if False:
            print('Hello World!')

        def overload_this(a):
            if False:
                while True:
                    i = 10
            return 'dummy'

        @njit
        def foo(a):
            if False:
                print('Hello World!')
            if a:
                s = 5
                s = overload_this(s)
            else:
                s = 'b'
            return s

        @overload(overload_this)
        def ol(a):
            if False:
                while True:
                    i = 10
            return overload_this
        self.check_func(foo, True)

    def test_issue3979(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(A, B):
            if False:
                while True:
                    i = 10
            x = A[0]
            y = B[0]
            for i in A:
                x = i
            for i in B:
                y = i
            return (x, y)
        self.check_func(foo, (1, 2), ('A', 'B'))

    def test_issue5219(self):
        if False:
            while True:
                i = 10

        def overload_this(a, b=None):
            if False:
                print('Hello World!')
            if isinstance(b, tuple):
                b = b[0]
            return b

        @overload(overload_this)
        def ol(a, b=None):
            if False:
                return 10
            b_is_tuple = isinstance(b, (types.Tuple, types.UniTuple))

            def impl(a, b=None):
                if False:
                    return 10
                if b_is_tuple is True:
                    b = b[0]
                return b
            return impl

        @njit
        def test_tuple(a, b):
            if False:
                while True:
                    i = 10
            overload_this(a, b)
        self.check_func(test_tuple, 1, (2,))

    def test_issue5223(self):
        if False:
            while True:
                i = 10

        @njit
        def bar(x):
            if False:
                for i in range(10):
                    print('nop')
            if len(x) == 5:
                return x
            x = x.copy()
            for i in range(len(x)):
                x[i] += 1
            return x
        a = np.ones(5)
        a.flags.writeable = False
        np.testing.assert_allclose(bar(a), bar.py_func(a))

    def test_issue5243(self):
        if False:
            print('Hello World!')

        @njit
        def foo(q):
            if False:
                for i in range(10):
                    print('nop')
            lin = np.array((0.1, 0.6, 0.3))
            stencil = np.zeros((3, 3))
            stencil[0, 0] = q[0, 0]
            return lin[0]
        self.check_func(foo, np.zeros((2, 2)))

    def test_issue5482_missing_variable_init(self):
        if False:
            while True:
                i = 10

        @njit('(intp, intp, intp)')
        def foo(x, v, n):
            if False:
                i = 10
                return i + 15
            for i in range(n):
                if i == 0:
                    if i == x:
                        pass
                    else:
                        problematic = v
                elif i == x:
                    pass
                else:
                    problematic = problematic + v
            return problematic

    def test_issue5482_objmode_expr_null_lowering(self):
        if False:
            return 10
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.untyped_passes import ReconstructSSA, IRProcessing
        from numba.core.typed_passes import PreLowerStripPhis

        class CustomPipeline(CompilerBase):

            def define_pipelines(self):
                if False:
                    for i in range(10):
                        print('nop')
                pm = DefaultPassBuilder.define_objectmode_pipeline(self.state)
                pm.add_pass_after(ReconstructSSA, IRProcessing)
                pm.add_pass_after(PreLowerStripPhis, ReconstructSSA)
                pm.finalize()
                return [pm]

        @jit('(intp, intp, intp)', looplift=False, pipeline_class=CustomPipeline)
        def foo(x, v, n):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(n):
                if i == n:
                    if i == x:
                        pass
                    else:
                        problematic = v
                elif i == x:
                    pass
                else:
                    problematic = problematic + v
            return problematic

    def test_issue5493_unneeded_phi(self):
        if False:
            while True:
                i = 10
        data = (np.ones(2), np.ones(2))
        A = np.ones(1)
        B = np.ones((1, 1))

        def foo(m, n, data):
            if False:
                print('Hello World!')
            if len(data) == 1:
                v0 = data[0]
            else:
                v0 = data[0]
                for _ in range(1, len(data)):
                    v0 += A
            for t in range(1, m):
                for idx in range(n):
                    t = B
                    if idx == 0:
                        if idx == n - 1:
                            pass
                        else:
                            problematic = t
                    elif idx == n - 1:
                        pass
                    else:
                        problematic = problematic + t
            return problematic
        expect = foo(10, 10, data)
        res1 = njit(foo)(10, 10, data)
        res2 = jit(forceobj=True, looplift=False)(foo)(10, 10, data)
        np.testing.assert_array_equal(expect, res1)
        np.testing.assert_array_equal(expect, res2)

    def test_issue5623_equal_statements_in_same_bb(self):
        if False:
            while True:
                i = 10

        def foo(pred, stack):
            if False:
                i = 10
                return i + 15
            i = 0
            c = 1
            if pred is True:
                stack[i] = c
                i += 1
                stack[i] = c
                i += 1
        python = np.array([0, 666])
        foo(True, python)
        nb = np.array([0, 666])
        njit(foo)(True, nb)
        expect = np.array([1, 1])
        np.testing.assert_array_equal(python, expect)
        np.testing.assert_array_equal(nb, expect)

    def test_issue5678_non_minimal_phi(self):
        if False:
            while True:
                i = 10
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.untyped_passes import ReconstructSSA, FunctionPass, register_pass
        phi_counter = []

        @register_pass(mutates_CFG=False, analysis_only=True)
        class CheckSSAMinimal(FunctionPass):
            _name = self.__class__.__qualname__ + '.CheckSSAMinimal'

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__(self)

            def run_pass(self, state):
                if False:
                    return 10
                ct = 0
                for blk in state.func_ir.blocks.values():
                    ct += len(list(blk.find_exprs('phi')))
                phi_counter.append(ct)
                return True

        class CustomPipeline(CompilerBase):

            def define_pipelines(self):
                if False:
                    print('Hello World!')
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(CheckSSAMinimal, ReconstructSSA)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=CustomPipeline)
        def while_for(n, max_iter=1):
            if False:
                return 10
            a = np.empty((n, n))
            i = 0
            while i <= max_iter:
                for j in range(len(a)):
                    for k in range(len(a)):
                        a[j, k] = j + k
                i += 1
            return a
        self.assertPreciseEqual(while_for(10), while_for.py_func(10))
        self.assertEqual(phi_counter, [1])

class TestSROAIssues(MemoryLeakMixin, TestCase):

    def test_issue7258_multiple_assignment_post_SSA(self):
        if False:
            print('Hello World!')
        cloned = []

        @register_pass(analysis_only=False, mutates_CFG=True)
        class CloneFoobarAssignments(FunctionPass):
            _name = 'clone_foobar_assignments_pass'

            def __init__(self):
                if False:
                    while True:
                        i = 10
                FunctionPass.__init__(self)

            def run_pass(self, state):
                if False:
                    i = 10
                    return i + 15
                mutated = False
                for blk in state.func_ir.blocks.values():
                    to_clone = []
                    for assign in blk.find_insts(ir.Assign):
                        if assign.target.name == 'foobar':
                            to_clone.append(assign)
                    for assign in to_clone:
                        clone = copy.deepcopy(assign)
                        blk.insert_after(clone, assign)
                        mutated = True
                        cloned.append(clone)
                return mutated

        class CustomCompiler(CompilerBase):

            def define_pipelines(self):
                if False:
                    while True:
                        i = 10
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state, 'custom_pipeline')
                pm._finalized = False
                pm.add_pass_after(CloneFoobarAssignments, ReconstructSSA)
                pm.add_pass_after(PreserveIR, NativeLowering)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=CustomCompiler)
        def udt(arr):
            if False:
                for i in range(10):
                    print('nop')
            foobar = arr + 1
            return foobar
        arr = np.arange(10)
        self.assertPreciseEqual(udt(arr), arr + 1)
        self.assertEqual(len(cloned), 1)
        self.assertEqual(cloned[0].target.name, 'foobar')
        nir = udt.overloads[udt.signatures[0]].metadata['preserved_ir']
        self.assertEqual(len(nir.blocks), 1, 'only one block')
        [blk] = nir.blocks.values()
        assigns = blk.find_insts(ir.Assign)
        foobar_assigns = [stmt for stmt in assigns if stmt.target.name == 'foobar']
        self.assertEqual(len(foobar_assigns), 2, "expected two assignment statements into 'foobar'")
        self.assertEqual(foobar_assigns[0], foobar_assigns[1], 'expected the two assignment statements to be the same')