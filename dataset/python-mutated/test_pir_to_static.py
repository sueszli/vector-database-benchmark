import os
import unittest
import numpy as np
import paddle

class TestDy2staticPir(unittest.TestCase):

    def test_basic_network(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                return 10
            out = paddle.mean(x)
            return out
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        ans = func(x)
        out = static_func(x)
        np.testing.assert_allclose(out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-08)

    def test_basic_network_backward(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                return 10
            out = paddle.mean(x)
            return out
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((3, 3))
        y = paddle.randn((3, 3))
        x.stop_gradient = False
        y.stop_gradient = False
        loss = func(x) * 2
        loss.backward()
        x_grad_ans = x.grad.numpy()
        x.clear_gradient()
        out = static_func(x)
        out = out * 2
        out.backward()
        st_grad = x.grad
        np.testing.assert_allclose(x_grad_ans, st_grad.numpy(), rtol=1e-05, atol=1e-08)

class TestDy2staticPir2(unittest.TestCase):

    def test_basic_layer(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.linear(x)
        net = SimpleNet()
        x = paddle.randn((10, 10))
        x.stop_gradient = False
        ans = net(x)
        net = paddle.jit.to_static(net, full_graph=True)
        out = net(x)
        np.testing.assert_allclose(out.numpy(), ans.numpy(), rtol=1e-05, atol=1e-08)

class TestDy2staticPir3(unittest.TestCase):

    def test_complex_layer(self):
        if False:
            for i in range(10):
                print('nop')

        def output_pure_func(x, y):
            if False:
                return 10
            outx = paddle.mean(x)
            outy = paddle.mean(y)
            outy.stop_gradient = True
            return (paddle.add(outx, outy), outy)

        def run_function(to_static=True):
            if False:
                for i in range(10):
                    print('nop')
            paddle.seed(2023)
            x = paddle.randn((10, 10))
            y = paddle.randn((10, 10))
            x.stop_gradient = False
            y.stop_gradient = True
            func = output_pure_func
            if to_static:
                func = paddle.jit.to_static(func, full_graph=True)
            (y, y_mean) = func(x, y)
            loss = y.mean()
            loss.backward()
            return (y, x.grad)
        for (dy, st) in zip(run_function(False), run_function(True)):
            np.testing.assert_allclose(dy.numpy(), st.numpy(), rtol=1e-05, atol=1e-08)

class TestLossFor10Steps(unittest.TestCase):

    def test_loss_for_10_steps(self):
        if False:
            return 10

        class SimpleNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x)

        def train_step(to_static=True):
            if False:
                for i in range(10):
                    print('nop')
            paddle.seed(2023)
            x = paddle.randn((10, 10), dtype='float32')
            y = paddle.randn((10, 10), dtype='float32')
            loss_fn = paddle.nn.loss.MSELoss()
            net = SimpleNet()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
            if to_static:
                net = paddle.jit.to_static(net, full_graph=True)
            losses = []
            for step in range(100):
                y_pred = net(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                losses.append(loss.numpy())
            return losses
        expected_losses = train_step(True)
        losses = train_step(False)
        np.testing.assert_allclose(losses, expected_losses, rtol=1e-05, atol=1e-08)

class TestDy2staticPir5(unittest.TestCase):

    def test_run(self):
        if False:
            while True:
                i = 10

        class SimpleNet(paddle.nn.Layer):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = paddle.nn.Linear(10, 10)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                if y is True:
                    return self.linear(x)
                else:
                    m = self.linear(x)
                    return m * m

        def train_step(to_static=True, full_graph=True):
            if False:
                i = 10
                return i + 15
            paddle.seed(2023)
            x = paddle.randn((10, 10), dtype='float32')
            y = paddle.randn((10, 10), dtype='float32')
            loss_fn = paddle.nn.loss.MSELoss()
            net = SimpleNet()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
            if to_static:
                net = paddle.jit.to_static(net, full_graph=full_graph)
            losses = []
            for step in range(100):
                y_pred = net(x, step % 2 == 1)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                losses.append(loss.numpy())
            return losses
        expected_losses = train_step(True)
        losses = train_step(False)
        np.testing.assert_allclose(losses, expected_losses, rtol=1e-05, atol=1e-08)
        os.environ['MIN_GRAPH_SIZE'] = '0'
        sot_losses = train_step(True, False)
        np.testing.assert_allclose(losses, sot_losses, rtol=1e-05, atol=1e-08)

class TestDy2staticPir6(unittest.TestCase):

    def test_basic_network(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                i = 10
                return i + 15
            shape = paddle.shape(x)
            out = shape[1:]
            return out
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((2, 3, 4))
        x.stop_gradient = False
        ans = func(x)
        out = static_func(x)
        np.testing.assert_allclose(out.numpy(), ans.numpy())

class TestDy2staticPir7(unittest.TestCase):

    def test_basic_network(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                i = 10
                return i + 15
            x = x * 2
            x = x + 1
            return 1
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((2, 3, 4))
        x.stop_gradient = False
        ans = func(x)
        out = static_func(x)
        np.testing.assert_allclose(out, ans)
if __name__ == '__main__':
    unittest.main()