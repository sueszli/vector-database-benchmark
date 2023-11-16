import unittest
import numpy
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle

def static_func(x, no_grad_x):
    if False:
        i = 10
        return i + 15
    tx = 2 * no_grad_x
    tx.stop_gradient = True
    return 2 * x

def main_func(x, index):
    if False:
        for i in range(10):
            print('nop')
    tmp = paddle.gather(x, index)
    out = paddle.jit.to_static(static_func)(x, tmp)
    return out

class TestNoGradientCase(Dy2StTestBase):

    def test_no_gradient(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        x = paddle.randn([10, 3])
        index = paddle.arange(0, 10, 1, dtype='int32')
        x.stop_gradient = False
        index.stop_gradient = True
        func = main_func
        output = func(x, index).mean()
        output.backward()
        self.assertTrue(x.grad is not None)
        self.assertTrue(numpy.all(x.grad.numpy() == paddle.full([10, 3], 2.0 / 30).numpy()))
        self.assertTrue(index.grad is None)
if __name__ == '__main__':
    unittest.main()