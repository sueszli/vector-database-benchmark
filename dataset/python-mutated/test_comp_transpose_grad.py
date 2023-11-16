import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core, framework

def apply_to_static(net, use_cinn):
    if False:
        for i in range(10):
            print('nop')
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out = paddle.transpose(x, [0, 2, 1])
        return out

@param.parameterized_class(('primal', 'axis', 'cotangent', 'dtype', 'rtol'), [(np.random.rand(100), [0], np.random.rand(100), np.float64, 1e-15), (np.random.rand(3, 4, 10), [0, 2, 1], np.random.rand(3, 10, 4), np.float64, 1e-15), (np.random.rand(2, 3, 4, 5), [0, 2, 3, 1], np.random.rand(2, 4, 5, 3), np.float64, 1e-15), (np.random.rand(2, 3, 4, 5, 6), [4, 2, 3, 1, 0], np.random.rand(6, 4, 5, 3, 2), np.float64, 1e-15), (np.random.rand(2, 3, 4, 5, 6, 1), [4, 2, 3, 1, 0, 5], np.random.rand(6, 4, 5, 3, 2, 1), np.float64, 1e-15), (np.random.rand(100), [0], np.random.rand(100), np.float16, 0.001), (np.random.rand(3, 4, 10), [0, 2, 1], np.random.rand(3, 10, 4), np.float16, 0.001), (np.random.rand(2, 3, 4, 5), [0, 2, 3, 1], np.random.rand(2, 4, 5, 3), np.float16, 0.001), (np.random.rand(2, 3, 4, 5, 6), [4, 2, 3, 1, 0], np.random.rand(6, 4, 5, 3, 2), np.float16, 0.001), (np.random.rand(2, 3, 4, 5, 6, 1), [4, 2, 3, 1, 0, 5], np.random.rand(6, 4, 5, 3, 2, 1), np.float16, 0.001)])
class TestTransposeGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        if isinstance(cls.primal, np.ndarray):
            cls.primal = cls.primal.astype(cls.dtype)
        if isinstance(cls.cotangent, np.ndarray):
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    def train(self, use_prim, use_cinn):
        if False:
            return 10
        paddle.seed(2022)
        self.x = paddle.randn([3, 4, 10])
        self.x.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x)
        res = paddle.autograd.grad(out, [self.x])
        return res

    def _test_cinn(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        use_cinn = True
        if isinstance(framework._current_expected_place(), framework.core.CPUPlace):
            use_cinn = False
        dy_res = self.train(use_prim=False, use_cinn=False)
        comp_st_cinn_res = self.train(use_prim=True, use_cinn=use_cinn)
        for i in range(len(dy_res)):
            np.testing.assert_allclose(comp_st_cinn_res[i].numpy(), dy_res[i].numpy(), rtol=1e-07, atol=1e-07)

    def test_transpose_grad_comp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

        def actual(primal, axis, cotangent):
            if False:
                return 10
            core._set_prim_backward_enabled(True)
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                if isinstance(primal, np.ndarray):
                    x = paddle.static.data('primal', primal.shape, primal.dtype)
                else:
                    x = paddle.static.data('primal', [1], 'float32')
                x.stop_gradient = False
                if isinstance(cotangent, np.ndarray):
                    v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                else:
                    v = paddle.static.data('cotangent', [1], 'float32')
                print(x.shape)
                y = paddle.transpose(x, axis)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent[0].name])[0]

        def desired(primal, axis, cotangent):
            if False:
                for i in range(10):
                    print('nop')
            core._set_prim_backward_enabled(False)
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                if isinstance(primal, np.ndarray):
                    x = paddle.static.data('primal', primal.shape, primal.dtype)
                else:
                    x = paddle.static.data('primal', [1], 'float32')
                x.stop_gradient = False
                if isinstance(cotangent, np.ndarray):
                    v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                else:
                    v = paddle.static.data('cotangent', [1], 'float32')
                y = paddle.transpose(x, axis)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent[0].name])[0]
        if self.dtype == np.float16 and isinstance(framework._current_expected_place(), framework.core.CPUPlace):
            pass
        else:
            np.testing.assert_allclose(actual=actual(self.primal, self.axis, self.cotangent), desired=desired(self.primal, self.axis, self.cotangent), rtol=self.rtol, atol=self.rtol)
        core._set_prim_backward_enabled(False)
if __name__ == '__main__':
    unittest.main()