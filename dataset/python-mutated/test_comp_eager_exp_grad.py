import unittest
import autograd
import autograd.numpy
import numpy as np
import parameterized as param
import paddle
from paddle.base import core

@param.parameterized_class(('primal', 'cotangent', 'dtype'), [(np.random.rand(10, 10), np.random.rand(10, 10), np.float32)])
class TestExpGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        core.set_prim_eager_enabled(True)
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        core.set_prim_eager_enabled(False)

    def test_exp_grad_comp(self):
        if False:
            for i in range(10):
                print('nop')

        def actual(primal, cotangent):
            if False:
                return 10
            primal = paddle.to_tensor(primal)
            primal.stop_gradient = False
            return paddle.grad(paddle.exp(primal), primal, paddle.to_tensor(cotangent))[0]

        def desired(primal, cotangent):
            if False:
                while True:
                    i = 10
            cotangent = np.ones_like(cotangent, dtype=primal.dtype) if cotangent is None else cotangent
            return autograd.make_vjp(autograd.numpy.exp)(primal)[0](cotangent)
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent), desired=desired(self.primal, self.cotangent), rtol=1e-06, atol=0)

    def test_stop_gradients(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            primal = paddle.to_tensor(self.primal)
            primal.stop_gradient = True
            return paddle.grad(paddle.exp(primal), primal, paddle.to_tensor(self.cotangent))
if __name__ == '__main__':
    unittest.main()