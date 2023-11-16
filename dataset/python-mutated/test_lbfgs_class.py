import unittest
import numpy as np
import paddle
from paddle.incubate.optimizer import lbfgs as incubate_lbfgs, line_search_dygraph
from paddle.optimizer import lbfgs
np.random.seed(123)

class Net(paddle.nn.Layer):

    def __init__(self, np_w, func):
        if False:
            return 10
        super().__init__()
        self.func = func
        w = paddle.to_tensor(np_w)
        self.w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.func(self.w, x)

def train_step(inputs, targets, net, opt):
    if False:
        for i in range(10):
            print('nop')

    def closure():
        if False:
            i = 10
            return i + 15
        outputs = net(inputs)
        loss = paddle.nn.functional.mse_loss(outputs, targets)
        opt.clear_grad()
        loss.backward()
        return loss
    loss = opt.step(closure)
    return loss

class TestLbfgs(unittest.TestCase):

    def test_function_fix_incubate(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        np_w = np.random.rand(1).astype(np.float32)
        input = np.random.rand(1).astype(np.float32)
        weights = [np.random.rand(1).astype(np.float32) for i in range(5)]
        targets = [weights[i] * input for i in range(5)]

        def func(w, x):
            if False:
                print('Hello World!')
            return w * x
        net = Net(np_w, func)
        opt = incubate_lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=5, line_search_fn='strong_wolfe', parameters=net.parameters())
        for (weight, target) in zip(weights, targets):
            input = paddle.to_tensor(input)
            target = paddle.to_tensor(target)
            loss = 1
            while loss > 0.0001:
                loss = train_step(input, target, net, opt)
            np.testing.assert_allclose(net.w, weight, rtol=1e-05)

    def test_inf_minima_incubate(self):
        if False:
            while True:
                i = 10
        input = np.random.rand(1).astype(np.float32)

        def outputs1(x):
            if False:
                while True:
                    i = 10
            return x * x * x - 3 * x * x + 3 * 1.01 * 0.99 * x

        def outputs2(x):
            if False:
                while True:
                    i = 10
            return pow(x, 4) + 5 * pow(x, 2)
        targets = [outputs1(input), outputs2(input)]
        input = paddle.to_tensor(input)

        def func1(extream_point, x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x * x - 3 * x * x + 3 * extream_point[0] * extream_point[1] * x

        def func2(extream_point, x):
            if False:
                return 10
            return pow(x, extream_point[0]) + 5 * pow(x, extream_point[1])
        extream_point = np.array([-2.34, 1.45]).astype('float32')
        net1 = Net(extream_point, func1)
        opt1 = incubate_lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe', parameters=net1.parameters())
        net2 = Net(extream_point, func2)
        opt2 = incubate_lbfgs.LBFGS(learning_rate=1, max_iter=50, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None, parameters=net2.parameters())
        n_iter = 0
        while n_iter < 20:
            loss = train_step(input, paddle.to_tensor(targets[0]), net1, opt1)
            n_iter = opt1.state_dict()['state']['func_evals']
        n_iter = 0
        while n_iter < 10:
            loss = train_step(input, paddle.to_tensor(targets[1]), net2, opt2)
            n_iter = opt1.state_dict()['state']['func_evals']

    def test_error_incubate(self):
        if False:
            print('Hello World!')

        def error_func1():
            if False:
                return 10
            extream_point = np.array([-1, 2]).astype('float32')
            extream_point = paddle.to_tensor(extream_point)
            return incubate_lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=3, line_search_fn='strong_wolfe', parameters=extream_point)
        self.assertRaises(TypeError, error_func1)

    def test_error2_incubate(self):
        if False:
            for i in range(10):
                print('nop')
        input = np.random.rand(1).astype(np.float32)

        def outputs2(x):
            if False:
                return 10
            return pow(x, 4) + 5 * pow(x, 2)
        targets = [outputs2(input)]
        input = paddle.to_tensor(input)

        def func2(extream_point, x):
            if False:
                while True:
                    i = 10
            return pow(x, extream_point[0]) + 5 * pow(x, extream_point[1])
        extream_point = np.array([-2.34, 1.45]).astype('float32')
        net2 = Net(extream_point, func2)
        opt2 = incubate_lbfgs.LBFGS(learning_rate=1, max_iter=50, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn='None', parameters=net2.parameters())

        def error_func():
            if False:
                i = 10
                return i + 15
            n_iter = 0
            while n_iter < 10:
                loss = train_step(input, paddle.to_tensor(targets[0]), net2, opt2)
                n_iter = opt2.state_dict()['state']['func_evals']
        self.assertRaises(RuntimeError, error_func)

    def test_line_search_incubate(self):
        if False:
            return 10

        def func1(x, alpha, d):
            if False:
                while True:
                    i = 10
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([0.0]))

        def func2(x, alpha, d):
            if False:
                while True:
                    i = 10
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([1.0]))

        def func3(x, alpha, d):
            if False:
                while True:
                    i = 10
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([-1.0]))
        line_search_dygraph._strong_wolfe(func1, paddle.to_tensor([1.0]), paddle.to_tensor([0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([0.0]), max_ls=1)
        line_search_dygraph._strong_wolfe(func1, paddle.to_tensor([1.0]), paddle.to_tensor([0.001]), paddle.to_tensor([0.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([0.0]), max_ls=0)
        line_search_dygraph._strong_wolfe(func2, paddle.to_tensor([1.0]), paddle.to_tensor([-0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), max_ls=1)
        line_search_dygraph._strong_wolfe(func3, paddle.to_tensor([1.0]), paddle.to_tensor([-0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), max_ls=1)
        line_search_dygraph._cubic_interpolate(paddle.to_tensor([2.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([1.0]), paddle.to_tensor([2.0]), paddle.to_tensor([0.0]), [0.1, 0.5])
        line_search_dygraph._cubic_interpolate(paddle.to_tensor([2.0]), paddle.to_tensor([0.0]), paddle.to_tensor([-3.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([-0.1]), [0.1, 0.5])

    def test_error3_incubate(self):
        if False:
            i = 10
            return i + 15

        def error_func3():
            if False:
                return 10
            extream_point = np.array([-1, 2]).astype('float32')
            extream_point = paddle.to_tensor(extream_point)

            def func(w, x):
                if False:
                    i = 10
                    return i + 15
                return w * x
            net = Net(extream_point, func)
            net.w = paddle.create_parameter(shape=[-1, 2], dtype=net.w.dtype)
            opt = incubate_lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=5, line_search_fn='strong_wolfe', parameters=net.parameters())
        self.assertRaises(AssertionError, error_func3)

    def test_function_fix(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        np_w = np.random.rand(1).astype(np.float32)
        input = np.random.rand(1).astype(np.float32)
        weights = [np.random.rand(1).astype(np.float32) for i in range(5)]
        targets = [weights[i] * input for i in range(5)]

        def func(w, x):
            if False:
                return 10
            return w * x
        net = Net(np_w, func)
        opt = lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=5, line_search_fn='strong_wolfe', parameters=net.parameters())
        for (weight, target) in zip(weights, targets):
            input = paddle.to_tensor(input)
            target = paddle.to_tensor(target)
            loss = 1
            while loss > 0.0001:
                loss = train_step(input, target, net, opt)
            np.testing.assert_allclose(net.w, weight, rtol=1e-05)

    def test_inf_minima(self):
        if False:
            return 10
        input = np.random.rand(1).astype(np.float32)

        def outputs1(x):
            if False:
                print('Hello World!')
            return x * x * x - 3 * x * x + 3 * 1.01 * 0.99 * x

        def outputs2(x):
            if False:
                i = 10
                return i + 15
            return pow(x, 4) + 5 * pow(x, 2)
        targets = [outputs1(input), outputs2(input)]
        input = paddle.to_tensor(input)

        def func1(extream_point, x):
            if False:
                print('Hello World!')
            return x * x * x - 3 * x * x + 3 * extream_point[0] * extream_point[1] * x

        def func2(extream_point, x):
            if False:
                print('Hello World!')
            return pow(x, extream_point[0]) + 5 * pow(x, extream_point[1])
        extream_point = np.array([-2.34, 1.45]).astype('float32')
        net1 = Net(extream_point, func1)
        opt1 = lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe', parameters=net1.parameters())
        net2 = Net(extream_point, func2)
        opt2 = lbfgs.LBFGS(learning_rate=1, max_iter=50, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None, parameters=net2.parameters())
        n_iter = 0
        while n_iter < 20:
            loss = train_step(input, paddle.to_tensor(targets[0]), net1, opt1)
            n_iter = opt1.state_dict()['state']['func_evals']
        n_iter = 0
        while n_iter < 10:
            loss = train_step(input, paddle.to_tensor(targets[1]), net2, opt2)
            n_iter = opt1.state_dict()['state']['func_evals']

    def test_error(self):
        if False:
            while True:
                i = 10

        def error_func1():
            if False:
                return 10
            extream_point = np.array([-1, 2]).astype('float32')
            extream_point = paddle.to_tensor(extream_point)
            return lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=3, line_search_fn='strong_wolfe', parameters=extream_point)
        self.assertRaises(TypeError, error_func1)

    def test_error2(self):
        if False:
            for i in range(10):
                print('nop')
        input = np.random.rand(1).astype(np.float32)

        def outputs2(x):
            if False:
                while True:
                    i = 10
            return pow(x, 4) + 5 * pow(x, 2)
        targets = [outputs2(input)]
        input = paddle.to_tensor(input)

        def func2(extream_point, x):
            if False:
                i = 10
                return i + 15
            return pow(x, extream_point[0]) + 5 * pow(x, extream_point[1])
        extream_point = np.array([-2.34, 1.45]).astype('float32')
        net2 = Net(extream_point, func2)
        opt2 = lbfgs.LBFGS(learning_rate=1, max_iter=50, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn='None', parameters=net2.parameters())

        def error_func():
            if False:
                while True:
                    i = 10
            n_iter = 0
            while n_iter < 10:
                loss = train_step(input, paddle.to_tensor(targets[0]), net2, opt2)
                n_iter = opt2.state_dict()['state']['func_evals']
        self.assertRaises(RuntimeError, error_func)

    def test_line_search(self):
        if False:
            return 10

        def func1(x, alpha, d):
            if False:
                print('Hello World!')
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([0.0]))

        def func2(x, alpha, d):
            if False:
                while True:
                    i = 10
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([1.0]))

        def func3(x, alpha, d):
            if False:
                i = 10
                return i + 15
            return (paddle.to_tensor(x + alpha * d), paddle.to_tensor([-1.0]))
        lbfgs._strong_wolfe(func1, paddle.to_tensor([1.0]), paddle.to_tensor([0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([0.0]), max_ls=1)
        lbfgs._strong_wolfe(func1, paddle.to_tensor([1.0]), paddle.to_tensor([0.001]), paddle.to_tensor([0.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([0.0]), max_ls=0)
        lbfgs._strong_wolfe(func2, paddle.to_tensor([1.0]), paddle.to_tensor([-0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), max_ls=1)
        lbfgs._strong_wolfe(func3, paddle.to_tensor([1.0]), paddle.to_tensor([-0.001]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), max_ls=1)
        lbfgs._cubic_interpolate(paddle.to_tensor([2.0]), paddle.to_tensor([1.0]), paddle.to_tensor([0.0]), paddle.to_tensor([1.0]), paddle.to_tensor([2.0]), paddle.to_tensor([0.0]), [0.1, 0.5])
        lbfgs._cubic_interpolate(paddle.to_tensor([2.0]), paddle.to_tensor([0.0]), paddle.to_tensor([-3.0]), paddle.to_tensor([1.0]), paddle.to_tensor([1.0]), paddle.to_tensor([-0.1]), [0.1, 0.5])

    def test_error3(self):
        if False:
            print('Hello World!')

        def error_func3():
            if False:
                for i in range(10):
                    print('nop')
            extream_point = np.array([-1, 2]).astype('float32')
            extream_point = paddle.to_tensor(extream_point)

            def func(w, x):
                if False:
                    i = 10
                    return i + 15
                return w * x
            net = Net(extream_point, func)
            net.w = paddle.create_parameter(shape=[-1, 2], dtype=net.w.dtype)
            opt = lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=5, line_search_fn='strong_wolfe', parameters=net.parameters())
        self.assertRaises(AssertionError, error_func3)

    def test_error4(self):
        if False:
            return 10
        paddle.disable_static()

        def error_func4():
            if False:
                while True:
                    i = 10
            inputs = np.random.rand(1).astype(np.float32)
            targets = paddle.to_tensor([inputs * 2])
            inputs = paddle.to_tensor(inputs)
            extream_point = np.array([-1, 1]).astype('float32')

            def func(extream_point, x):
                if False:
                    print('Hello World!')
                return x * extream_point[0] + 5 * x * extream_point[1]
            net = Net(extream_point, func)
            opt = lbfgs.LBFGS(learning_rate=1, max_iter=10, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=5, line_search_fn='strong_wolfe', parameters=net.parameters())
            loss = train_step(inputs, targets, net, opt)
            opt.minimize(loss)
        self.assertRaises(NotImplementedError, error_func4)
if __name__ == '__main__':
    unittest.main()