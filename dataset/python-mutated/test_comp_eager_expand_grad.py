import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core

@param.parameterized_class(('name', 'primal', 'cotangent', 'shape', 'dtype'), (('same_shape', np.random.rand(10, 10), np.random.rand(10, 10), (10, 10), np.float32), ('same_rank', np.random.rand(1, 10), np.random.rand(10, 10), (10, 10), np.float32), ('same_rank', np.random.rand(10, 1, 10, 1), np.random.rand(10, 10, 10, 10), (10, 10, 10, 10), np.float32), ('diff_rank', np.random.rand(1, 10, 1), np.random.rand(10, 10, 10, 10), (10, 10, 10, 10), np.float32)))
class TestExpandGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        core.set_prim_eager_enabled(False)

    def test_comp(self):
        if False:
            for i in range(10):
                print('nop')

        def func(primal, cotangent, shape):
            if False:
                for i in range(10):
                    print('nop')
            primal = paddle.to_tensor(primal)
            primal.stop_gradient = False
            cotangent = paddle.to_tensor(cotangent)
            return paddle.grad(paddle.expand(primal, shape), primal, cotangent)[0]

        def actual(primal, cotangent, shape):
            if False:
                while True:
                    i = 10
            core.set_prim_eager_enabled(True)
            return func(primal, cotangent, shape)

        def desired(primal, cotangent, shape):
            if False:
                while True:
                    i = 10
            core.set_prim_eager_enabled(False)
            return func(primal, cotangent, shape)
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, self.shape), desired=desired(self.primal, self.cotangent, self.shape), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()