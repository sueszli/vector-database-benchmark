import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core, framework

@param.parameterized_class(('name', 'primals', 'stop_gradients', 'cotangents', 'dtype'), (('test_normal_case', (np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)), (False, False), (np.random.rand(2, 3, 4),), np.float32), ('test_broadcast_diff_rank', (np.random.rand(2, 3, 1, 4), np.random.rand(3, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_broadcast_same_rank', (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_stop_gradient', (np.random.rand(2, 3, 1, 4), np.random.rand(2, 1, 3, 4)), (False, True), (np.random.rand(2, 3, 3, 4),), np.float32), ('test_reduce_axe_empty', (np.random.rand(2, 3, 3, 4), np.random.rand(2, 1, 3, 4)), (False, False), (np.random.rand(2, 3, 3, 4),), np.float32)))
class TestMultiplyGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.primals = tuple((primal.astype(cls.dtype) for primal in cls.primals))
        cls.cotangents = tuple((co.astype(cls.dtype) for co in cls.cotangents))

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def as_tuple(self, x):
        if False:
            return 10
        return (x,) if isinstance(x, framework.Variable) else x

    def vjp(self):
        if False:
            return 10
        (primals, cotangents) = (self.primals, self.cotangents)
        (mp, sp) = (paddle.static.Program(), paddle.static.Program())
        with paddle.static.program_guard(mp, sp):
            primals = tuple((paddle.static.data(f'primal{i}', primal.shape, primal.dtype) for (i, primal) in enumerate(primals)))
            for (primal, flag) in zip(primals, self.stop_gradients):
                primal.stop_gradient = flag
            cotangents = tuple((paddle.static.data(f'cotangent{i}', co.shape, co.dtype) for (i, co) in enumerate(cotangents)))
            out = self.as_tuple(paddle.multiply(*primals))
            grads = paddle.static.gradients(out, primals, cotangents)
        exe = paddle.static.Executor()
        exe.run(sp)
        return exe.run(program=mp, feed={**{f'primal{i}': primal for (i, primal) in enumerate(self.primals)}, **{f'cotangent{i}': co for (i, co) in enumerate(self.cotangents)}}, fetch_list=[g for g in grads if g is not None])

    def test_comp(self):
        if False:
            return 10
        core._set_prim_backward_enabled(True)
        actual = self.vjp()
        core._set_prim_backward_enabled(False)
        desired = self.vjp()
        self.assertEqual(len(actual), len(desired))
        for (i, j) in zip(actual, desired):
            np.testing.assert_allclose(i, j, rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()