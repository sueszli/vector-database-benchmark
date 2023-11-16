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
            while True:
                i = 10
        core.set_prim_eager_enabled(True)
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def test_sigmoid_grad_comp(self):
        if False:
            while True:
                i = 10

        def actual(primal, cotangent):
            if False:
                while True:
                    i = 10
            core._set_prim_backward_enabled(True)
            paddle.enable_static()
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                dout = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                x.stop_gradient = False
                res = F.sigmoid(x)
                x_grad = paddle.static.gradients(res, [x], dout)
                exe = paddle.static.Executor()
                exe.run(sp)
                out = exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_grad[0].name])
            return out[0]

        def desired(primal, cotangent):
            if False:
                for i in range(10):
                    print('nop')
            core._set_prim_backward_enabled(False)
            paddle.enable_static()
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                dout = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                x.stop_gradient = False
                res = F.sigmoid(x)
                x_grad = paddle.static.gradients(res, [x], dout)
                exe = paddle.static.Executor()
                exe.run(sp)
                out = exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_grad[0].name])
            return out[0]
        np.testing.assert_allclose(actual=actual(self.primal, self.cotangent), desired=desired(self.primal, self.cotangent), rtol=1e-06, atol=0)
if __name__ == '__main__':
    unittest.main()