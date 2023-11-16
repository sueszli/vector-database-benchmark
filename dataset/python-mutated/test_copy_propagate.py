from numba import jit, njit
from numba.core import types, typing, ir, config, compiler, cpu
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import copy_propagate, apply_copy_propagate, get_name_var_table
from numba.core.typed_passes import type_inference_stage
from numba.tests.support import IRPreservingTestPipeline
import numpy as np
import unittest

def test_will_propagate(b, z, w):
    if False:
        return 10
    x = 3
    x1 = x
    if b > 0:
        y = z + w
    else:
        y = 0
    a = 2 * x1
    return a < b

def test_wont_propagate(b, z, w):
    if False:
        i = 10
        return i + 15
    x = 3
    if b > 0:
        y = z + w
        x = 1
    else:
        y = 0
    a = 2 * x
    return a < b

def null_func(a, b, c, d):
    if False:
        while True:
            i = 10
    False

def inListVar(list_var, var):
    if False:
        print('Hello World!')
    for i in list_var:
        if i.name == var:
            return True
    return False

def findAssign(func_ir, var):
    if False:
        i = 10
        return i + 15
    for (label, block) in func_ir.blocks.items():
        for (i, inst) in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name != var:
                all_var = inst.list_vars()
                if inListVar(all_var, var):
                    return True
    return False

class TestCopyPropagate(unittest.TestCase):

    def test1(self):
        if False:
            return 10
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx, 'cpu')
        test_ir = compiler.run_frontend(test_will_propagate)
        with cpu_target.nested_context(typingctx, targetctx):
            typingctx.refresh()
            targetctx.refresh()
            args = (types.int64, types.int64, types.int64)
            (typemap, return_type, calltypes, _) = type_inference_stage(typingctx, targetctx, test_ir, args, None)
            type_annotation = type_annotations.TypeAnnotation(func_ir=test_ir, typemap=typemap, calltypes=calltypes, lifted=(), lifted_from=None, args=args, return_type=return_type, html_output=config.HTML)
            (in_cps, out_cps) = copy_propagate(test_ir.blocks, typemap)
            apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)
            self.assertFalse(findAssign(test_ir, 'x1'))

    def test2(self):
        if False:
            for i in range(10):
                print('nop')
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx, 'cpu')
        test_ir = compiler.run_frontend(test_wont_propagate)
        with cpu_target.nested_context(typingctx, targetctx):
            typingctx.refresh()
            targetctx.refresh()
            args = (types.int64, types.int64, types.int64)
            (typemap, return_type, calltypes, _) = type_inference_stage(typingctx, targetctx, test_ir, args, None)
            type_annotation = type_annotations.TypeAnnotation(func_ir=test_ir, typemap=typemap, calltypes=calltypes, lifted=(), lifted_from=None, args=args, return_type=return_type, html_output=config.HTML)
            (in_cps, out_cps) = copy_propagate(test_ir.blocks, typemap)
            apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)
            self.assertTrue(findAssign(test_ir, 'x'))

    def test_input_ir_extra_copies(self):
        if False:
            while True:
                i = 10
        'make sure Interpreter._remove_unused_temporaries() has removed extra copies\n        in the IR in simple cases so copy propagation is faster\n        '

        def test_impl(a):
            if False:
                while True:
                    i = 10
            b = a + 3
            return b
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(test_impl(5), j_func(5))
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        self.assertTrue(len(fir.blocks) == 1)
        block = next(iter(fir.blocks.values()))
        b_found = False
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and stmt.target.name == 'b':
                b_found = True
                self.assertTrue(isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop' and (stmt.value.lhs.name == 'a'))
        self.assertTrue(b_found)

    def test_input_ir_copy_remove_transform(self):
        if False:
            print('Hello World!')
        'make sure Interpreter._remove_unused_temporaries() does not generate\n        invalid code for rare chained assignment cases\n        '

        def impl1(a):
            if False:
                for i in range(10):
                    print('nop')
            b = c = a + 1
            return (b, c)

        def impl2(A, i, a):
            if False:
                print('Hello World!')
            b = A[i] = a + 1
            return (b, A[i] + 2)

        def impl3(A, a):
            if False:
                print('Hello World!')
            b = A.a = a + 1
            return (b, A.a + 2)

        class C:
            pass
        self.assertEqual(impl1(5), njit(impl1)(5))
        self.assertEqual(impl2(np.ones(3), 0, 5), njit(impl2)(np.ones(3), 0, 5))
        self.assertEqual(impl3(C(), 5), jit(forceobj=True)(impl3)(C(), 5))
if __name__ == '__main__':
    unittest.main()