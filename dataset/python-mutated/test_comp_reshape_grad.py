import unittest
import numpy as np
import parameterized as param
import paddle
from paddle.base import core, framework

def apply_to_static(net, use_cinn):
    if False:
        while True:
            i = 10
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        tmp = self.fc(x)
        out = paddle.reshape(tmp, [2, 1, 4])
        return out

@param.parameterized_class(('primal', 'shape', 'cotangent', 'dtype', 'rtol'), [(np.random.rand(10, 1, 10), [10, 10], np.random.rand(10, 10), np.float32, 1e-05), (np.random.rand(2, 60), [12, 10], np.random.rand(12, 10), np.float32, 1e-05), (np.random.rand(10, 1, 10), [10, 10], np.random.rand(10, 10), np.float64, 1e-15), (np.random.rand(2, 60), [12, 10], np.random.rand(12, 10), np.float64, 1e-15), (np.random.rand(10, 1, 10), [10, 10], np.random.rand(10, 10), np.float16, 0.001), (np.random.rand(2, 60), [12, 10], np.random.rand(12, 10), np.float16, 0.001)])
class TestReshapeGradComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.primal = cls.primal.astype(cls.dtype)
        cls.cotangent = cls.cotangent.astype(cls.dtype)

    def train(self, use_prim, use_cinn):
        if False:
            print('Hello World!')
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False
        net = PrimeNet()
        core._set_prim_backward_enabled(use_prim)
        net = apply_to_static(net, use_cinn)
        out = net(self.x)
        res = paddle.autograd.grad(out, [self.x])
        return res

    def test_cinn(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        use_cinn = True
        if isinstance(framework._current_expected_place(), framework.core.CPUPlace):
            use_cinn = False
        dy_res = self.train(use_prim=False, use_cinn=False)
        comp_st_cinn_res = self.train(use_prim=True, use_cinn=use_cinn)
        for i in range(len(dy_res)):
            np.testing.assert_allclose(comp_st_cinn_res[i].numpy(), dy_res[i].numpy(), rtol=1e-07, atol=1e-07)
        paddle.enable_static()

    def test_reshape_grad_comp(self):
        if False:
            print('Hello World!')

        def actual(primal, shape, cotangent):
            if False:
                i = 10
                return i + 15
            core._set_prim_backward_enabled(True)
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                y = paddle.reshape(x, shape)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent[0].name])[0]

        def desired(primal, shape, cotangent):
            if False:
                print('Hello World!')
            core._set_prim_backward_enabled(False)
            (mp, sp) = (paddle.static.Program(), paddle.static.Program())
            with paddle.static.program_guard(mp, sp):
                x = paddle.static.data('primal', primal.shape, primal.dtype)
                x.stop_gradient = False
                v = paddle.static.data('cotangent', cotangent.shape, cotangent.dtype)
                y = paddle.reshape(x, shape)
                x_cotangent = paddle.static.gradients(y, x, v)
            exe = paddle.static.Executor()
            exe.run(sp)
            return exe.run(program=mp, feed={'primal': primal, 'cotangent': cotangent}, fetch_list=[x_cotangent[0].name])[0]
        if self.dtype == np.float16 and isinstance(framework._current_expected_place(), framework.core.CPUPlace):
            pass
        else:
            np.testing.assert_allclose(actual=actual(self.primal, self.shape, self.cotangent), desired=desired(self.primal, self.shape, self.cotangent), rtol=self.rtol, atol=self.rtol)
        core._set_prim_backward_enabled(False)
if __name__ == '__main__':
    unittest.main()