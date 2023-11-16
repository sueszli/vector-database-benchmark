import unittest
import numpy as np
import paddle
from paddle import base

class TestAdamaxAPI(unittest.TestCase):

    def test_adamax_api_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype('float32')
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.Adamax(learning_rate=0.01, parameters=linear.parameters(), weight_decay=0.01)
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_adamax_api(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        place = base.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = base.Executor(place)
        train_prog = base.Program()
        startup = base.Program()
        with base.program_guard(train_prog, startup):
            with base.unique_name.guard():
                data = paddle.static.data(name='data', shape=shape)
                conv = paddle.static.nn.conv2d(data, 8, 3)
                loss = paddle.mean(conv)
                beta1 = 0.85
                beta2 = 0.95
                opt = paddle.optimizer.Adamax(learning_rate=1e-05, beta1=beta1, beta2=beta2, weight_decay=0.01, epsilon=1e-08)
                opt.minimize(loss)
        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={'data': data_np}, fetch_list=[loss])
        assert rets[0] is not None

class TestAdamaxAPIGroup(TestAdamaxAPI):

    def test_adamax_api_dygraph(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype('float32')
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        adam = paddle.optimizer.Adamax(learning_rate=0.01, parameters=[{'params': linear_1.parameters()}, {'params': linear_2.parameters(), 'weight_decay': 0.001, 'beta1': 0.1, 'beta2': 0.99}], weight_decay=0.1)
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()
if __name__ == '__main__':
    unittest.main()