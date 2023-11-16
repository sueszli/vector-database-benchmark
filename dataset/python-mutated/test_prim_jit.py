import unittest
import numpy as np
import paddle
from paddle.framework import core

def func(x):
    if False:
        for i in range(10):
            print('nop')
    x1 = paddle.mean(x)
    out = paddle.nn.functional.gelu(x1, False)
    return out

class TestDy2staticPir(unittest.TestCase):

    def test_basic_network_backward(self):
        if False:
            for i in range(10):
                print('nop')
        core._set_prim_all_enabled(True)
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((8, 16, 64))
        x.stop_gradient = False
        ref_out = func(x) * 2
        ref_out.backward()
        ref_grad = x.grad.numpy()
        x.clear_gradient()
        out = static_func(x)
        actual_out = out * 2
        actual_out.backward()
        actual_grad = x.grad
        core._set_prim_all_enabled(False)
        ops = [op.name() for op in static_func.program_cache.last()[-1][-1].train_program.program.global_block().ops]
        assert 'pd_op.erf' in ops
        assert 'pd_op.gelu' not in ops
        np.testing.assert_allclose(ref_out, actual_out.numpy(), atol=1e-06, rtol=1e-06)
        np.testing.assert_allclose(ref_grad, actual_grad.numpy(), atol=1e-06, rtol=1e-06)

class TestDy2staticPirEval(unittest.TestCase):

    def test_basic_network_backward_(self):
        if False:
            while True:
                i = 10
        core._set_prim_all_enabled(True)
        static_func = paddle.jit.to_static(func, full_graph=True)
        static_func.eval()
        x = paddle.randn((8, 16, 64))
        x.stop_gradient = False
        ref_out = func(x) * 2
        out = static_func(x)
        actual_out = out * 2
        ops = [op.name() for op in static_func.program_cache.last()[-1][-1].infer_program.program.global_block().ops]
        core._set_prim_all_enabled(False)
        assert 'pd_op.erf' in ops
        assert 'pd_op.gelu' not in ops
        np.testing.assert_allclose(ref_out, actual_out.numpy(), atol=1e-06, rtol=1e-06)
if __name__ == '__main__':
    unittest.main()