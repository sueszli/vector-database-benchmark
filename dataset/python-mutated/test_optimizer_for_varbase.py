import unittest
import numpy as np
import paddle
from paddle import optimizer

class TestOptimizerForVarBase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.lr = 0.01

    def run_optimizer_step_with_varbase_list_input(self, optimizer):
        if False:
            print('Hello World!')
        x = paddle.zeros([2, 3])
        y = paddle.ones([2, 3])
        x.stop_gradient = False
        z = x + y
        opt = optimizer(learning_rate=self.lr, parameters=[x], weight_decay=0.01)
        z.backward()
        opt.step()
        np.testing.assert_allclose(x.numpy(), np.full([2, 3], -self.lr), rtol=1e-05)

    def run_optimizer_minimize_with_varbase_list_input(self, optimizer):
        if False:
            print('Hello World!')
        x = paddle.zeros([2, 3])
        y = paddle.ones([2, 3])
        x.stop_gradient = False
        z = x + y
        opt = optimizer(learning_rate=self.lr, parameters=[x])
        z.backward()
        opt.minimize(z)
        np.testing.assert_allclose(x.numpy(), np.full([2, 3], -self.lr), rtol=1e-05)

    def test_adam_with_varbase_list_input(self):
        if False:
            return 10
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adam)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adam)

    def test_sgd_with_varbase_list_input(self):
        if False:
            while True:
                i = 10
        self.run_optimizer_step_with_varbase_list_input(optimizer.SGD)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.SGD)

    def test_adagrad_with_varbase_list_input(self):
        if False:
            print('Hello World!')
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adagrad)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adagrad)

    def test_adamw_with_varbase_list_input(self):
        if False:
            print('Hello World!')
        self.run_optimizer_step_with_varbase_list_input(optimizer.AdamW)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.AdamW)

    def test_adamax_with_varbase_list_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_optimizer_step_with_varbase_list_input(optimizer.Adamax)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Adamax)

    def test_momentum_with_varbase_list_input(self):
        if False:
            i = 10
            return i + 15
        self.run_optimizer_step_with_varbase_list_input(optimizer.Momentum)
        self.run_optimizer_minimize_with_varbase_list_input(optimizer.Momentum)

    def test_optimizer_with_varbase_input(self):
        if False:
            while True:
                i = 10
        x = paddle.zeros([2, 3])
        with self.assertRaises(TypeError):
            optimizer.Adam(learning_rate=self.lr, parameters=x)

    def test_create_param_lr_with_1_for_coverage(self):
        if False:
            while True:
                i = 10
        x = paddle.base.framework.EagerParamBase(dtype='float32', shape=[5, 10], lod_level=0, name='x', optimize_attr={'learning_rate': 1.0})
        x.value().get_tensor().set(np.random.random((5, 10)).astype('float32'), paddle.base.framework._current_expected_place())
        y = paddle.ones([5, 10])
        z = x + y
        opt = optimizer.Adam(learning_rate=self.lr, parameters=[x])
        z.backward()
        opt.step()

    def test_create_param_lr_with_no_1_value_for_coverage(self):
        if False:
            return 10
        x = paddle.base.framework.EagerParamBase(dtype='float32', shape=[5, 10], lod_level=0, name='x', optimize_attr={'learning_rate': 0.12})
        x.value().get_tensor().set(np.random.random((5, 10)).astype('float32'), paddle.base.framework._current_expected_place())
        y = paddle.ones([5, 10])
        z = x + y
        opt = optimizer.Adam(learning_rate=self.lr, parameters=[x])
        z.backward()
        opt.step()
if __name__ == '__main__':
    unittest.main()