"""
Tests for practical lowering specific errors.
"""
import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase

def issue7507_lround(a):
    if False:
        i = 10
        return i + 15
    'Dummy function used in test'
    pass

class TestLowering(MemoryLeakMixin, TestCase):

    def test_issue4156_loop_vars_leak(self):
        if False:
            for i in range(10):
                print('nop')
        "Test issues with zero-filling of refct'ed variables inside loops.\n\n        Before the fix, the in-loop variables are always zero-filled at their\n        definition location. As a result, their state from the previous\n        iteration is erased. No decref is applied. To fix this, the\n        zero-filling must only happen once after the alloca at the function\n        entry block. The loop variables are technically defined once per\n        function (one alloca per definition per function), but semantically\n        defined once per assignment. Semantically, their lifetime stop only\n        when the variable is re-assigned or when the function ends.\n        "

        @njit
        def udt(N):
            if False:
                i = 10
                return i + 15
            sum_vec = np.zeros(3)
            for n in range(N):
                if n >= 0:
                    vec = np.ones(1)
                if n >= 0:
                    sum_vec += vec[0]
            return sum_vec
        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant1(self):
        if False:
            return 10
        'Variant of test_issue4156_loop_vars_leak.\n\n        Adding an outer loop.\n        '

        @njit
        def udt(N):
            if False:
                print('Hello World!')
            sum_vec = np.zeros(3)
            for x in range(N):
                for y in range(N):
                    n = x + y
                    if n >= 0:
                        vec = np.ones(1)
                    if n >= 0:
                        sum_vec += vec[0]
            return sum_vec
        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant2(self):
        if False:
            for i in range(10):
                print('nop')
        'Variant of test_issue4156_loop_vars_leak.\n\n        Adding deeper outer loop.\n        '

        @njit
        def udt(N):
            if False:
                i = 10
                return i + 15
            sum_vec = np.zeros(3)
            for z in range(N):
                for x in range(N):
                    for y in range(N):
                        n = x + y + z
                        if n >= 0:
                            vec = np.ones(1)
                        if n >= 0:
                            sum_vec += vec[0]
            return sum_vec
        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant3(self):
        if False:
            i = 10
            return i + 15
        'Variant of test_issue4156_loop_vars_leak.\n\n        Adding inner loop around allocation\n        '

        @njit
        def udt(N):
            if False:
                while True:
                    i = 10
            sum_vec = np.zeros(3)
            for z in range(N):
                for x in range(N):
                    n = x + z
                    if n >= 0:
                        for y in range(N):
                            vec = np.ones(y)
                    if n >= 0:
                        sum_vec += vec[0]
            return sum_vec
        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue4156_loop_vars_leak_variant4(self):
        if False:
            print('Hello World!')
        'Variant of test_issue4156_loop_vars_leak.\n\n        Interleaves loops and allocations\n        '

        @njit
        def udt(N):
            if False:
                for i in range(10):
                    print('nop')
            sum_vec = 0
            for n in range(N):
                vec = np.zeros(7)
                for n in range(N):
                    z = np.zeros(7)
                sum_vec += vec[0] + z[0]
            return sum_vec
        got = udt(4)
        expect = udt.py_func(4)
        self.assertPreciseEqual(got, expect)

    def test_issue_with_literal_in_static_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an issue with literal type used as index of static_getitem\n        '

        @register_pass(mutates_CFG=False, analysis_only=False)
        class ForceStaticGetitemLiteral(FunctionPass):
            _name = 'force_static_getitem_literal'

            def __init__(self):
                if False:
                    return 10
                FunctionPass.__init__(self)

            def run_pass(self, state):
                if False:
                    while True:
                        i = 10
                repl = {}
                for (inst, sig) in state.calltypes.items():
                    if isinstance(inst, ir.Expr) and inst.op == 'static_getitem':
                        [obj, idx] = sig.args
                        new_sig = sig.replace(args=(obj, types.literal(inst.index)))
                        repl[inst] = new_sig
                state.calltypes.update(repl)
                return True

        class CustomPipeline(CompilerBase):

            def define_pipelines(self):
                if False:
                    print('Hello World!')
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(ForceStaticGetitemLiteral, NopythonTypeInference)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=CustomPipeline)
        def foo(arr):
            if False:
                for i in range(10):
                    print('nop')
            return arr[4]
        arr = np.arange(10)
        got = foo(arr)
        expect = foo.py_func(arr)
        self.assertEqual(got, expect)

    def test_issue7507(self):
        if False:
            while True:
                i = 10
        '\n        Test a problem with BaseContext.get_function() because of changes\n        related to the new style error handling.\n        '
        from numba.core.typing.templates import AbstractTemplate, infer_global
        from numba.core.imputils import lower_builtin

        @infer_global(issue7507_lround)
        class lroundTemplate(AbstractTemplate):
            key = issue7507_lround

            def generic(self, args, kws):
                if False:
                    while True:
                        i = 10
                signature = types.int64(types.float64)

                @lower_builtin(issue7507_lround, types.float64)
                def codegen(context, builder, sig, args):
                    if False:
                        print('Hello World!')
                    return context.cast(builder, args[0], sig.args[0], sig.return_type)
                return signature

        @njit('int64(float64)')
        def foo(a):
            if False:
                print('Hello World!')
            return issue7507_lround(a)
        self.assertEqual(foo(3.4), 3)