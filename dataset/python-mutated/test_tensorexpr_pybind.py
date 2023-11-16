import torch
import numpy as np
import torch._C._te as te
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest
LLVM_ENABLED = torch._C._llvm_enabled()

def construct_adder(n: int, dtype=torch.float32):
    if False:
        print('Hello World!')
    A = te.BufHandle('A', [n], dtype)
    B = te.BufHandle('B', [n], dtype)

    def compute(i):
        if False:
            for i in range(10):
                print('nop')
        return A.load([i]) + B.load([i])
    C = te.Compute('C', [n], compute)
    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    return te.construct_codegen('ir_eval', stmt, [A, B, C])

class TestTensorExprPyBind(JitTestCase):

    def test_simple_sum(self):
        if False:
            return 10
        n = 32
        cg = construct_adder(n)
        tA = torch.randn(n)
        tB = torch.randn(n)
        tC = torch.empty(n)
        cg.call([tA, tB, tC])
        torch.testing.assert_close(tA + tB, tC)

    def test_call_raw(self):
        if False:
            for i in range(10):
                print('nop')
        n = 16
        cg = construct_adder(n, dtype=torch.float64)
        tA = torch.randn(n, dtype=torch.float64)
        tB = torch.randn(n, dtype=torch.float64)
        tC = torch.empty(n, dtype=torch.float64)
        cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])
        torch.testing.assert_close(tA + tB, tC)

    def test_external_calls(self):
        if False:
            for i in range(10):
                print('nop')
        dtype = torch.float32
        A = te.BufHandle('A', [1, 4], dtype)
        B = te.BufHandle('B', [4, 1], dtype)
        C = te.BufHandle('C', [1, 1], dtype)
        s = te.ExternalCall(C, 'nnc_aten_matmul', [A, B], [])
        loopnest = te.LoopNest(s, [C])
        loopnest.prepare_for_codegen()
        codegen = te.construct_codegen('ir_eval', s, [A, B, C])
        tA = torch.ones(1, 4)
        tB = torch.ones(4, 1)
        tC = torch.empty(1, 1)
        codegen.call([tA, tB, tC])
        torch.testing.assert_close(torch.matmul(tA, tB), tC)

    def test_dynamic_shape(self):
        if False:
            i = 10
            return i + 15
        dN = te.VarHandle(torch.int32)
        A = te.BufHandle([dN], torch.float64)
        B = te.BufHandle([dN], torch.float64)

        def compute(i):
            if False:
                return 10
            return A.load(i) - B.load(i)
        C = te.Compute('C', [dN], compute)
        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()
        cg = te.construct_codegen('ir_eval', loopnest.simplify(), [A, B, C, dN])

        def test_with_shape(n):
            if False:
                i = 10
                return i + 15
            tA = torch.randn(n, dtype=torch.double)
            tB = torch.randn(n, dtype=torch.double)
            tC = torch.empty(n, dtype=torch.double)
            cg.call([tA, tB, tC, n])
            torch.testing.assert_close(tA - tB, tC)
        test_with_shape(8)
        test_with_shape(31)

    def test_dynamic_shape_2d(self):
        if False:
            return 10
        dN = te.VarHandle(torch.int32)
        dM = te.VarHandle(torch.int32)
        A = te.BufHandle([dN, dM], torch.float64)
        B = te.BufHandle([dN, dM], torch.float64)

        def compute(i, j):
            if False:
                for i in range(10):
                    print('nop')
            return A.load([i, j]) - B.load([i, j])
        C = te.Compute('C', [dN, dM], compute)
        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()
        cg = te.construct_codegen('ir_eval', loopnest.simplify(), [A, B, C, dN, dM])

        def test_with_shape(n, m):
            if False:
                for i in range(10):
                    print('nop')
            tA = torch.randn(n, m, dtype=torch.double)
            tB = torch.randn(n, m, dtype=torch.double)
            tC = torch.empty(n, m, dtype=torch.double)
            cg.call([tA, tB, tC, n, m])
            torch.testing.assert_close(tA - tB, tC)
        test_with_shape(2, 4)
        test_with_shape(5, 3)

    def test_dtype_error(self):
        if False:
            return 10
        te.BufHandle('a', [1], torch.float32)
        self.assertRaises(TypeError, lambda : te.BufHandle('a', [1], 'float55'))

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_tensor_inputs(self):
        if False:
            return 10

        def f(a, b, c):
            if False:
                return 10
            return a + b + c
        (device, size) = ('cpu', (4, 4))
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)
        graph_str = '\ngraph(%a.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),\n      %b.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),\n      %c.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu)):\n  %6 : int = prim::Constant[value=1]()\n  %7 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %6)\n  %3 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%7, %c.1, %6)\n  return (%3)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_scalar_inputs(self):
        if False:
            return 10

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a + b + c
        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(0.6, dtype=torch.float, device='cpu')
        z = torch.tensor(0.7, dtype=torch.float, device='cpu')
        graph_str = '\ngraph(%a.1 : Float(requires_grad=0, device=cpu),\n      %b.1 : Float(requires_grad=0, device=cpu),\n      %c.1 : Float(requires_grad=0, device=cpu)):\n  %3 : int = prim::Constant[value=1]()\n  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)\n  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)\n  return (%9)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_shape_prop(self):
        if False:
            for i in range(10):
                print('nop')
        (device, size) = ('cpu', (4, 4))
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        graph_str = '\ngraph(%a : Tensor, %b : Tensor):\n  %c : Tensor = aten::mul(%a, %b)\n  return (%c)\n        '
        graph = torch._C.parse_ir(graph_str)
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
            pass
        assert exception_thrown
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        kernel = te.TensorExprKernel(graph)
        res = kernel.run((x, y))
        correct = torch.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-05)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_shape_prop_module(self):
        if False:
            return 10

        class TestModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x * x + y
        graph = torch.jit.script(TestModule()).graph
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
            pass
        assert exception_thrown
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        exception_thrown = False
        try:
            torch._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            exception_thrown = True
            pass
        assert exception_thrown
        torch._C._te.remove_unused_self_argument(graph)
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        kernel = te.TensorExprKernel(graph)
        (device, size) = ('cpu', (4, 4))
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-05)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_t(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                print('Hello World!')
            return a.t()
        (device, size) = ('cpu', (3, 4))
        x = torch.rand(size, device=device)
        graph_str = '\ngraph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):\n  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)\n  return (%3)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_transpose(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                i = 10
                return i + 15
            return a.transpose(-1, -2)
        (device, size) = ('cpu', (3, 4))
        x = torch.rand(size, device=device)
        graph_str = '\ngraph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):\n  %2 : int = prim::Constant[value=-1]()\n  %3 : int = prim::Constant[value=-2]()\n  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)\n  return (%4)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_permute(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a.permute([2, 1, 0])
        (device, size) = ('cpu', (3, 4, 5))
        x = torch.rand(size, device=device)
        graph_str = '\ngraph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):\n  %1 : int = prim::Constant[value=2]()\n  %2 : int = prim::Constant[value=1]()\n  %3 : int = prim::Constant[value=0]()\n  %4 : int[] = prim::ListConstruct(%1, %2, %3)\n  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)\n  return (%5)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_custom_lowering(self):
        if False:
            return 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a.nan_to_num()
        device = 'cpu'
        x = torch.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = torch.nan
        graph_str = '\ngraph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):\n    %none : NoneType = prim::Constant()\n    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)\n    return (%y)\n        '
        graph = torch._C.parse_ir(graph_str)

        def my_custom_lowering(inputs, out_shape, out_stride, out_type, device):
            if False:
                print('Hello World!')

            def compute(idxs):
                if False:
                    i = 10
                    return i + 15
                load = inputs[0].as_buf().load(idxs)
                return te.ifThenElse(te.ExprHandle.isnan(load), te.ExprHandle.float(0.0), load)
            return te.Compute2('custom_nan_to_num', out_shape, compute)
        kernel = te.TensorExprKernel(graph, {'aten::nan_to_num': my_custom_lowering})
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_kernel_with_expand(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            return a.expand((2, 3, 4))
        device = 'cpu'
        x = torch.rand((1, 3, 1), device=device)
        graph_str = '\ngraph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):\n  %1 : int = prim::Constant[value=2]()\n  %2 : int = prim::Constant[value=3]()\n  %3 : int = prim::Constant[value=4]()\n  %4 : int[] = prim::ListConstruct(%1, %2, %3)\n  %5 : bool = prim::Constant[value=0]()\n  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)\n  return (%6)\n        '
        graph = torch._C.parse_ir(graph_str)
        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=0.002)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=0.002)

    @unittest.skipIf(not LLVM_ENABLED, 'LLVM backend not enabled')
    def test_alloc_in_loop(self):
        if False:
            return 10
        (a, tmp, b) = (te.BufHandle(name, [1], torch.float32) for name in ['a', 'tmp', 'b'])
        body = te.Block([tmp.store([0], a.load([0])), b.store([0], tmp.load([0]))])
        for _ in range(4):
            i = te.VarHandle('i', torch.int32)
            body = te.For.make(i, 0, 100, body)
        nest = te.LoopNest(body, [b])
        nest.prepare_for_codegen()
        f = te.construct_codegen('llvm', nest.simplify(), [a, b])
        (ta, tb) = (torch.ones(1) for _ in range(2))
        f.call([ta.data_ptr(), tb.data_ptr()])

class TestExprHandlePyBind(JitTestCase):

    def test_unary_ops(self):
        if False:
            print('Hello World!')
        unary_operators = {torch.sin: torch._C._te.sin, torch.cos: torch._C._te.cos, torch.tan: torch._C._te.tan, torch.asin: torch._C._te.asin, torch.acos: torch._C._te.acos, torch.atan: torch._C._te.atan, torch.sinh: torch._C._te.sinh, torch.cosh: torch._C._te.cosh, torch.tanh: torch._C._te.tanh, torch.sigmoid: torch._C._te.sigmoid, torch.exp: torch._C._te.exp, torch.expm1: torch._C._te.expm1, torch.abs: torch._C._te.abs, torch.log: torch._C._te.log, torch.log2: torch._C._te.log2, torch.log10: torch._C._te.log10, torch.log1p: torch._C._te.log1p, torch.erf: torch._C._te.erf, torch.erfc: torch._C._te.erfc, torch.sqrt: torch._C._te.sqrt, torch.rsqrt: torch._C._te.rsqrt, torch.ceil: torch._C._te.ceil, torch.floor: torch._C._te.floor, torch.round: torch._C._te.round, torch.trunc: torch._C._te.trunc, torch.lgamma: torch._C._te.lgamma, torch.frac: torch._C._te.frac}

        def construct_te_fn(op, n: int, dtype=torch.float32):
            if False:
                for i in range(10):
                    print('nop')
            A = torch._C._te.BufHandle('A', [n], dtype)

            def compute(i):
                if False:
                    i = 10
                    return i + 15
                return op(A.load([i]))
            C = te.Compute('C', [n], compute)
            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = te.simplify(loopnest.root_stmt())
            return te.construct_codegen('ir_eval', stmt, [A, C])
        n = 10
        a = torch.rand(n)
        for (torch_op, te_op) in unary_operators.items():
            ref = torch_op(a)
            te_fn = construct_te_fn(te_op, n, torch.float32)
            res = torch.empty(n)
            te_fn.call([a, res])
            assert torch.allclose(ref, res, atol=0.001, rtol=0.001)
if __name__ == '__main__':
    run_tests()