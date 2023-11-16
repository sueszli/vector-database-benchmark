import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core

@param.parameterized_class(('primal', 'cotangent', 'src_dtype', 'dst_type'), [(np.random.rand(10, 10), np.random.rand(10, 10), np.float32, np.float64), (np.random.rand(10, 10), np.random.rand(10, 10), np.float64, np.float32), (np.random.rand(10, 10), np.random.rand(10, 10), np.float32, np.float32)])
class TestCastGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.primal = cls.primal.astype(cls.src_dtype)
        cls.cotangent = cls.cotangent.astype(cls.src_dtype)

    def test_cast_grad_comp(self):
        if False:
            for i in range(10):
                print('nop')
        core.set_prim_eager_enabled(True)

        def actual(primal, cotangent):
            if False:
                while True:
                    i = 10
            x = paddle.to_tensor(primal)
            x.stop_gradient = False
            v = paddle.to_tensor(cotangent)
            y = paddle.cast(x, self.dst_type)
            x_cotangent = paddle.grad(y, x, v)
            return x_cotangent

        def desired(primal, cotangent):
            if False:
                for i in range(10):
                    print('nop')
            return (cotangent * np.ones_like(primal)).astype(primal.dtype)
        actual = actual(self.primal, self.cotangent)
        desired = desired(self.primal, self.cotangent)
        from paddle.base.data_feeder import _PADDLE_DTYPE_2_NUMPY_DTYPE
        self.assertEqual(_PADDLE_DTYPE_2_NUMPY_DTYPE[actual[0].dtype], desired.dtype)
        np.testing.assert_allclose(actual=actual[0], desired=desired, rtol=1e-06, atol=0)
        core.set_prim_eager_enabled(False)
if __name__ == '__main__':
    unittest.main()