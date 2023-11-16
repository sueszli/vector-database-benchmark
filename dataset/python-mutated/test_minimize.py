import unittest
import numpy as np
import paddle
from paddle.incubate.autograd.primx import prim2orig
from paddle.incubate.autograd.utils import disable_prim, enable_prim, prim_enabled
paddle.enable_static()

class TestMinimize(unittest.TestCase):

    def model(self, x, w, bias, opt):
        if False:
            while True:
                i = 10
        paddle.seed(0)
        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input_x = paddle.static.data('x', x.shape, dtype=x.dtype)
            input_x.stop_gradient = False
            params_w = paddle.static.create_parameter(shape=w.shape, dtype=w.dtype, is_bias=False)
            params_bias = paddle.static.create_parameter(shape=bias.shape, dtype=bias.dtype, is_bias=True)
            y = paddle.tanh(paddle.matmul(input_x, params_w) + params_bias)
            loss = paddle.norm(y, p=2)
            opt = opt
            (_, grads) = opt.minimize(loss)
            if prim_enabled():
                prim2orig(main.block(0))
        exe.run(startup)
        grads = exe.run(main, feed={'x': x, 'w': w, 'bias': bias}, fetch_list=grads)
        return grads

    def test_adam(self):
        if False:
            return 10
        x = np.random.rand(2, 20)
        w = np.random.rand(20, 2)
        bias = np.random.rand(2)
        enable_prim()
        prim_grads = self.model(x, w, bias, paddle.optimizer.Adam(0.01))
        disable_prim()
        orig_grads = self.model(x, w, bias, paddle.optimizer.Adam(0.01))
        for (orig, prim) in zip(orig_grads, prim_grads):
            np.testing.assert_allclose(orig, prim)

    def test_sgd(self):
        if False:
            while True:
                i = 10
        x = np.random.rand(2, 20)
        w = np.random.rand(20, 2)
        bias = np.random.rand(2)
        enable_prim()
        prim_grads = self.model(x, w, bias, paddle.optimizer.SGD(0.01))
        disable_prim()
        orig_grads = self.model(x, w, bias, paddle.optimizer.SGD(0.01))
        for (orig, prim) in zip(orig_grads, prim_grads):
            np.testing.assert_allclose(orig, prim)
if __name__ == '__main__':
    unittest.main()