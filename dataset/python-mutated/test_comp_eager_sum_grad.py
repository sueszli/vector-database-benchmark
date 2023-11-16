import unittest
import numpy as np
import paddle
from paddle.base import core

def actual(primal, cotangent, axis, keep_dim):
    if False:
        print('Hello World!')
    core.set_prim_eager_enabled(False)
    x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
    v = paddle.to_tensor(cotangent, dtype='float32', stop_gradient=False)
    y = paddle.sum(x, axis=axis, keepdim=keep_dim)
    x_cotangent = paddle.grad(y, x, v, create_graph=True, retain_graph=True)
    return x_cotangent[0]

def desired(primal, cotangent, axis, keep_dim):
    if False:
        return 10
    core.set_prim_eager_enabled(True)
    x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
    v = paddle.to_tensor(cotangent, dtype='float32', stop_gradient=False)
    y = paddle.sum(x, axis=axis, keepdim=keep_dim)
    x_cotangent = paddle.grad(y, x, v, create_graph=True, retain_graph=True)
    return x_cotangent[0]

class TestSumGradComp(unittest.TestCase):

    def test_sum_grad_comp_1(self):
        if False:
            return 10
        self.primal = np.random.rand(10, 10)
        self.cotangent = np.array(np.random.rand())
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [], False), desired=desired(self.primal, self.cotangent, [], False), rtol=1e-06, atol=0)

    def test_sum_grad_comp_2(self):
        if False:
            return 10
        self.primal = np.random.rand(4, 3, 2)
        self.cotangent = np.random.rand(4, 2)
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, 1, False), desired=desired(self.primal, self.cotangent, 1, False), rtol=1e-06, atol=0)

    def test_sum_grad_comp_3(self):
        if False:
            return 10
        self.primal = np.random.rand(4, 3, 2)
        self.cotangent = np.random.rand(4, 1, 2)
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, 1, True), desired=desired(self.primal, self.cotangent, 1, True), rtol=1e-06, atol=0)

    def test_sum_grad_comp_4(self):
        if False:
            while True:
                i = 10
        self.primal = np.random.rand(4, 3, 2, 5)
        self.cotangent = np.random.rand(4, 1, 2, 1)
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [1, 3], True), desired=desired(self.primal, self.cotangent, [1, 3], True), rtol=1e-06, atol=0)

    def test_sum_grad_comp_5(self):
        if False:
            print('Hello World!')
        self.primal = np.random.rand(4, 3, 2, 5)
        self.cotangent = np.random.rand(4, 2)
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [1, 3], False), desired=desired(self.primal, self.cotangent, [1, 3], False), rtol=1e-06, atol=0)

    def test_sum_grad_comp_6(self):
        if False:
            return 10
        self.primal = np.random.rand(3, 2, 5)
        self.cotangent = np.random.rand(3, 1, 1)
        paddle.disable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [-2, -1], True), desired=desired(self.primal, self.cotangent, [-2, -1], True), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()