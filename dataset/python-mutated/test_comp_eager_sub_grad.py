import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core
core.set_prim_eager_enabled(True)

@param.parameterized_class(('primal0', 'primal1', 'dtype'), [(np.random.rand(2, 3, 4), np.random.rand(2, 3, 4), np.float32), (np.random.rand(2, 3, 3, 4), np.random.rand(3, 1, 4), np.float32), (np.random.rand(2, 3, 3, 4), np.random.rand(2, 3, 1, 4), np.float32), (np.random.rand(2, 3, 3, 4), np.random.rand(2, 3, 1, 4), np.float32), (np.random.rand(2, 3, 3, 4), np.random.rand(2, 3, 1, 1), np.float32)])
class TestSubGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.primal0 = cls.primal0.astype(cls.dtype)
        cls.primal1 = cls.primal1.astype(cls.dtype)

    def test_sub_grad_comp(self):
        if False:
            while True:
                i = 10

        def actual(primal0, primal1):
            if False:
                print('Hello World!')
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.subtract(x, y)
            res = paddle.grad(out, [x, y], create_graph=True, retain_graph=True)
            return (res[0].numpy(), res[1].numpy())

        def desired(primal0, primal1):
            if False:
                i = 10
                return i + 15
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            y = paddle.to_tensor(primal1, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.subtract(x, y)
            res = paddle.grad(out, [x, y], create_graph=True, retain_graph=True)
            return (res[0].numpy(), res[1].numpy())
        (dx, dy) = actual(self.primal0, self.primal1)
        (ddx, ddy) = desired(self.primal0, self.primal1)
        np.testing.assert_allclose(actual=dx, desired=ddx, rtol=1e-06, atol=0)
        np.testing.assert_allclose(actual=dy, desired=ddy, rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()