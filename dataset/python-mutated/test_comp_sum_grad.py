import unittest
import numpy as np
import paddle
from paddle.base import core

def actual(primal, cotangent, axis, keep_dim):
    if False:
        print('Hello World!')
    core._set_prim_backward_enabled(False)
    (mp, sp) = (paddle.static.Program(), paddle.static.Program())
    with paddle.static.program_guard(mp, sp):
        x = paddle.static.data('primal', primal.shape, primal.dtype)
        x.stop_gradient = False
        v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
        y = paddle.sum(x, axis=axis, keepdim=keep_dim)
        x_cotangent = paddle.static.gradients(y, x, None)
    exe = paddle.static.Executor()
    exe.run(sp)
    result = exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent])[0]
    return result

def desired(primal, cotangent, axis, keep_dim):
    if False:
        for i in range(10):
            print('nop')
    core._set_prim_backward_enabled(True)
    (mp, sp) = (paddle.static.Program(), paddle.static.Program())
    with paddle.static.program_guard(mp, sp):
        x = paddle.static.data('primal', primal.shape, primal.dtype)
        x.stop_gradient = False
        v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
        y = paddle.sum(x, axis=axis, keepdim=keep_dim)
        x_cotangent = paddle.static.gradients(y, x, None)
    exe = paddle.static.Executor()
    exe.run(sp)
    result = exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent])[0]
    return result

class TestSumGradComp(unittest.TestCase):

    def test_sum_grad_comp_1(self):
        if False:
            print('Hello World!')
        self.primal = np.random.rand(10, 10)
        self.cotangent = np.random.rand(1, 1)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [], True), desired=desired(self.primal, self.cotangent, [], True), rtol=1e-06, atol=0)

    def test_sum_grad_comp_2(self):
        if False:
            return 10
        self.primal = np.random.rand(4, 3, 2)
        self.cotangent = np.random.rand(4, 2)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, 1, False), desired=desired(self.primal, self.cotangent, 1, False), rtol=1e-06, atol=0)

    def test_sum_grad_comp_3(self):
        if False:
            while True:
                i = 10
        self.primal = np.random.rand(4, 3, 2)
        self.cotangent = np.random.rand(4, 1, 2)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, 1, True), desired=desired(self.primal, self.cotangent, 1, True), rtol=1e-06, atol=0)

    def test_sum_grad_comp_4(self):
        if False:
            i = 10
            return i + 15
        self.primal = np.random.rand(4, 3, 2, 5)
        self.cotangent = np.random.rand(4, 1, 2, 1)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [1, 3], True), desired=desired(self.primal, self.cotangent, [1, 3], True), rtol=1e-06, atol=0)

    def test_sum_grad_comp_5(self):
        if False:
            for i in range(10):
                print('nop')
        self.primal = np.random.rand(4, 3, 2, 5)
        self.cotangent = np.random.rand(4, 2)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [1, 3], False), desired=desired(self.primal, self.cotangent, [1, 3], False), rtol=1e-06, atol=0)

    def test_sum_grad_comp_6(self):
        if False:
            print('Hello World!')
        self.primal = np.random.rand(3, 2, 5)
        self.cotangent = np.random.rand(3, 1, 1)
        paddle.enable_static()
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, [-2, -1], True), desired=desired(self.primal, self.cotangent, [-2, -1], True), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()