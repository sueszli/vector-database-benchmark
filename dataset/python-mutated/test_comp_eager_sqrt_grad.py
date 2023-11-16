import unittest
import autograd
import autograd.numpy
import numpy as np
import parameterized as param
import paddle
from paddle.base import core
core.set_prim_eager_enabled(True)

@param.parameterized_class(('primal', 'cotangent', 'dtype'), [(np.random.rand(10, 10), np.random.rand(10, 10), np.float32)])
class TestSqrtGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    def test_sqrt_grad_comp(self):
        if False:
            print('Hello World!')

        def actual(primal, cotangent):
            if False:
                return 10
            paddle.disable_static()
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            v = paddle.to_tensor(cotangent, dtype='float32', stop_gradient=False)
            y = paddle.sqrt(x)
            return paddle.grad(y, x, v, create_graph=True, retain_graph=True)[0]

        def desired(primal, cotangent):
            if False:
                while True:
                    i = 10
            return autograd.make_vjp(autograd.numpy.sqrt)(primal)[0](cotangent)
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent), desired=desired(self.primal, self.cotangent), rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()