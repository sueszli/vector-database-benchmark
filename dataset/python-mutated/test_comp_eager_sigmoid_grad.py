import unittest
import numpy as np
import parameterized as param
import paddle
import paddle.nn.functional as F
from paddle.base import core

@param.parameterized_class(('primal', 'cotangent', 'dtype'), [(np.random.rand(10, 10), np.random.rand(10, 10), np.float32)])
class TestExpGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        core.set_prim_eager_enabled(True)
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def test_sigmoid_grad_comp(self):
        if False:
            print('Hello World!')

        def actual(primal, cotangent):
            if False:
                return 10
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal)
            dout = paddle.to_tensor(cotangent)
            x.stop_gradient = False
            return paddle.grad(F.sigmoid(x), x, dout)[0]

        def desired(primal, cotangent):
            if False:
                print('Hello World!')
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal)
            dout = paddle.to_tensor(cotangent)
            x.stop_gradient = False
            return paddle.grad(F.sigmoid(x), x, dout)[0]
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent), desired=desired(self.primal, self.cotangent), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()