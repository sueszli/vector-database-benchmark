import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core

@param.parameterized_class(('name', 'primal', 'cotangent', 'shape', 'dtype'), (('same_shape', np.random.rand(10, 10), np.random.rand(10, 10), (10, 10), np.float32), ('same_rank', np.random.rand(1, 10), np.random.rand(10, 10), (10, 10), np.float32), ('same_rank', np.random.rand(10, 1, 10, 1), np.random.rand(10, 10, 10, 10), (10, 10, 10, 10), np.float32), ('diff_rank', np.random.rand(1, 10, 1), np.random.rand(10, 10, 10, 10), (10, 10, 10, 10), np.float32), ('single_direction_broadcast', np.random.rand(10, 10, 10, 10), np.random.rand(1, 10, 1), (10, 10, 10, 10), np.float32)))
class TestExpandGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)
        paddle.enable_static()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        core._set_prim_backward_enabled(False)

    def test_comp(self):
        if False:
            for i in range(10):
                print('nop')

        def func(primal, cotangent, shape):
            if False:
                for i in range(10):
                    print('nop')
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                y = paddle.expand(x, shape)
                x_cotangent = paddle.static.gradients(y, x)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=x_cotangent)[0]

        def actual(primal, cotangent, shape):
            if False:
                return 10
            core._set_prim_backward_enabled(True)
            return func(primal, cotangent, shape)

        def desired(primal, cotangent, shape):
            if False:
                for i in range(10):
                    print('nop')
            core._set_prim_backward_enabled(False)
            return func(primal, cotangent, shape)
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent, self.shape), desired=desired(self.primal, self.cotangent, self.shape), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()