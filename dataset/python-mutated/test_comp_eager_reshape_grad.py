import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core
core.set_prim_eager_enabled(True)

@param.parameterized_class(('primal', 'shape', 'cotangent', 'dtype'), [(np.random.rand(10, 1, 10), [10, 10], np.random.rand(10, 10), np.float32), (np.random.rand(2, 60), [12, 10], np.random.rand(12, 10), np.float32)])
class TestReshapeGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.primal = cls.primal.astype(cls.dtype)

    def test_reshape_grad_comp(self):
        if False:
            i = 10
            return i + 15

        def actual(primal0, shape):
            if False:
                return 10
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()

        def desired(primal0, shape):
            if False:
                i = 10
                return i + 15
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.reshape(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()
        dx = actual(self.primal, self.shape)
        ddx = desired(self.primal, self.shape)
        np.testing.assert_allclose(actual=dx, desired=ddx, rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()