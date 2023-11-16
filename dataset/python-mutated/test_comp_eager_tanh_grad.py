import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core
core.set_prim_eager_enabled(True)

@param.parameterized_class(('primal', 'dtype'), [(np.random.rand(2, 3, 4), np.float32), (np.random.rand(2, 3, 3, 4), np.float32)])
class TestTanhGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.primal = cls.primal.astype(cls.dtype)

    def test_tanh_grad_comp(self):
        if False:
            for i in range(10):
                print('nop')

        def actual(primal):
            if False:
                i = 10
                return i + 15
            paddle.disable_static()
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.tanh(x)
            x_cotangent = paddle.grad(y, x, create_graph=True, retain_graph=True)
            return x_cotangent[0]

        def desired(primal):
            if False:
                while True:
                    i = 10
            paddle.disable_static()
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.tanh(x)
            x_cotangent = paddle.grad(y, x, create_graph=True, retain_graph=True)
            return x_cotangent[0]
        np.testing.assert_allclose(actual=actual(self.primal), desired=desired(self.primal), rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()