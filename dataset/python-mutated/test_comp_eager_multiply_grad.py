import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core

@param.parameterized_class(('name', 'primals', 'stop_gradients', 'cotangents', 'dtype'), (('test_normal_case', (np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)), (False, False), (np.random.rand(2, 3, 4),), np.float32), ('test_broadcast_diff_rank', (np.random.rand(2, 3, 1, 4), np.random.rand(3, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_broadcast_same_rank', (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_stop_gradient', (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)), (False, True), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_reduce_axe_empty', (np.random.rand(2, 3, 3, 4), np.random.rand(2, 1, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32)))
class TestMultiplyGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.primals = tuple((primal.astype(cls.dtype) for primal in cls.primals))
        cls.cotangents = tuple((co.astype(cls.dtype) for co in cls.cotangents))

    def as_tuple(self, x):
        if False:
            print('Hello World!')
        return (x,) if isinstance(x, paddle.Tensor) else x

    def vjp(self):
        if False:
            i = 10
            return i + 15
        (primals, cotangents) = (self.primals, self.cotangents)
        primals = tuple((paddle.to_tensor(primal) for primal in primals))
        for (primal, flag) in zip(primals, self.stop_gradients):
            primal.stop_gradient = flag
        cotangents = tuple((paddle.to_tensor(co) for co in cotangents))
        out = self.as_tuple(paddle.multiply(*primals))
        grads = paddle.grad(out, primals, cotangents, allow_unused=True)
        return [g for g in grads if g is not None]

    def test_comp(self):
        if False:
            for i in range(10):
                print('nop')
        core.set_prim_eager_enabled(True)
        actual = self.vjp()
        core.set_prim_eager_enabled(False)
        desired = self.vjp()
        for (i, j) in zip(actual, desired):
            np.testing.assert_allclose(i, j, rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()