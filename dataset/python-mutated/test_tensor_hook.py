import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle
from paddle import nn
from paddle.jit import to_static

class TestStaticAnalysis(Dy2StTestBase):

    def test_hook_for_different_parameter(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15

            def h(g):
                if False:
                    print('Hello World!')
                return 2 * g
            y = x + 4
            f = y + x
            z = f ** 2
            y.register_hook(h)
            f.register_hook(h)
            x.register_hook(h)
            return z
        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()
        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    def test_hook_for_reassignment_parameter(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')

            def h(g):
                if False:
                    print('Hello World!')
                return 2 * g
            y = x + 4
            x = y * 5
            z = x ** 2
            x.register_hook(h)
            return z
        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()
        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    def test_hook_for_repeat_register(self):
        if False:
            return 10

        def f(x):
            if False:
                print('Hello World!')

            def h(g):
                if False:
                    return 10
                return 2 * g
            y = x + 4
            z = y ** 2
            x.register_hook(h)
            x.register_hook(h)
            return z
        x = paddle.to_tensor([2.0])
        x.stop_gradient = False
        loss = f(x)
        loss.backward()
        x_jit = paddle.to_tensor([2.0])
        x_jit.stop_gradient = False
        jit_f = to_static(f)
        loss = jit_f(x_jit)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), x_jit.grad.numpy())

    @test_legacy_and_pir_exe_and_pir_api
    def test_hook_in_init_for_layer(self):
        if False:
            for i in range(10):
                print('nop')

        def hook(grad):
            if False:
                return 10
            return grad * 2
        IMAGE_SIZE = 784
        CLASS_NUM = 10

        class LinearNet(nn.Layer):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                self._linear.parameters()[0].register_hook(hook)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self._linear(x)
        layer = LinearNet()
        jit_layer = to_static(LinearNet())
        data = np.random.random([IMAGE_SIZE]).astype('float32')
        image = paddle.to_tensor(data)
        image_jit = paddle.to_tensor(data)
        loss = layer(image)
        loss_jit = jit_layer(image_jit)
        loss_jit.backward()
        loss.backward()
        np.testing.assert_allclose(layer.parameters()[0].grad.numpy(), jit_layer.parameters()[0].grad.numpy())
if __name__ == '__main__':
    unittest.main()