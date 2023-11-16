import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core
core.set_prim_eager_enabled(True)

@param.parameterized_class(('primal', 'axis', 'cotangent', 'dtype'), [(np.random.rand(100), [0], np.random.rand(100), np.float32), (np.random.rand(3, 4, 10), [0, 2, 1], np.random.rand(3, 10, 4), np.float32), (np.random.rand(2, 3, 4, 5), [0, 2, 3, 1], np.random.rand(2, 4, 5, 3), np.float32), (np.random.rand(2, 3, 4, 5, 6), [4, 2, 3, 1, 0], np.random.rand(6, 4, 5, 3, 2), np.float32), (np.random.rand(2, 3, 4, 5, 6, 1), [4, 2, 3, 1, 0, 5], np.random.rand(6, 4, 5, 3, 2, 1), np.float32)])
class TestTransposeGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        if isinstance(cls.primal, np.ndarray):
            cls.primal = cls.primal.astype(cls.dtype)

    def test_transpose_grad_comp(self):
        if False:
            print('Hello World!')

        def actual(primal0, shape):
            if False:
                print('Hello World!')
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.transpose(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()

        def desired(primal0, shape):
            if False:
                while True:
                    i = 10
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(primal0, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            out = paddle.transpose(x, shape)
            res = paddle.grad(out, [x], create_graph=True, retain_graph=True)
            return res[0].numpy()
        dx = actual(self.primal, self.axis)
        ddx = desired(self.primal, self.axis)
        np.testing.assert_allclose(actual=dx, desired=ddx, rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()