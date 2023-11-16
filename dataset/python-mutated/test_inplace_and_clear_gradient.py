import unittest
import numpy as np
import paddle
from paddle import _legacy_C_ops
paddle.disable_static()

def clear_grad(w, a):
    if False:
        return 10

    @paddle.no_grad()
    def warp(*_):
        if False:
            i = 10
            return i + 15
        assert w.grad is not None
        _legacy_C_ops.scale_(w.grad, 'scale', 0.5)
        w.clear_gradient(False)
    return warp

class TestInplaceAndClearGradient(unittest.TestCase):

    def test(self):
        if False:
            print('Hello World!')
        paddle.set_device('cpu')
        input_data = np.ones([2, 2]).astype('float32')
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        _clear_grad = clear_grad(w, a='1')
        w._register_backward_hook(_clear_grad)
        for i in range(10):
            out = _legacy_C_ops.scale(w, 'scale', 0.1)
            out.backward()
if __name__ == '__main__':
    unittest.main()