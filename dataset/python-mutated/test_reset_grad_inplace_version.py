import unittest
import numpy as np
import paddle
from paddle import _legacy_C_ops
paddle.set_device('cpu')

def clear_grad_test_0(w, a):
    if False:
        while True:
            i = 10

    @paddle.no_grad()
    def warp(*_):
        if False:
            return 10
        assert w.grad is not None
        _legacy_C_ops.scale_(w.grad, 'scale', 0.5)
        w._reset_grad_inplace_version(True)
    return warp

class TestInplaceAndClearGradient(unittest.TestCase):

    def test_inplace_n_clear_grad(self):
        if False:
            while True:
                i = 10
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        _clear_grad = clear_grad_test_0(w, a='1')
        w._register_backward_hook(_clear_grad)
        for i in range(2):
            print(' Step: ', i)
            out0 = _legacy_C_ops.scale(w, 'scale', 0.1)
            out = _legacy_C_ops.matmul_v2(out0, w, 'trans_x', False, 'trans_y', False)
            out.backward()
        assert w.grad[0] == 0.15

class Counter:

    def __init__(self):
        if False:
            return 10
        self.num_calls = 0
        self.step = 0

def clear_grad_test_1(w, c):
    if False:
        while True:
            i = 10

    @paddle.no_grad()
    def warp(*_):
        if False:
            i = 10
            return i + 15
        assert w.grad is not None
        if c.step == 1:
            w.grad.scale_(scale=0.5)
            w._reset_grad_inplace_version(True)
        c.num_calls += 1
    return warp

class TestInplaceClearGradAccumulation(unittest.TestCase):

    def test_inplace_clear_grad_accum(self):
        if False:
            for i in range(10):
                print('nop')
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        c = Counter()
        _clear_grad = clear_grad_test_1(w, c)
        w._register_backward_hook(_clear_grad)
        for c.step in range(5):
            out0 = _legacy_C_ops.scale(w, 'scale', 0.1)
            out = _legacy_C_ops.matmul_v2(out0, w, 'trans_x', False, 'trans_y', False)
            out.backward()
            if c.step == 1:
                w.clear_gradient(False)
            assert c.num_calls == 1
            c.num_calls = 0

class TestInplaceClearGradAccumulationAlt(unittest.TestCase):

    def test_inplace_clear_grad_accum(self):
        if False:
            print('Hello World!')
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        out = _legacy_C_ops.scale(w, 'scale', 0.1)
        out.backward()
        w.grad.scale_(scale=0.5)
        w._reset_grad_inplace_version(False)
        assert w.grad._inplace_version() == 1
if __name__ == '__main__':
    unittest.main()