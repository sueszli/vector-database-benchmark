import unittest
import numpy as np
import paddle
from paddle import nn, static
paddle.enable_static()

class SimpleNet(nn.Layer):

    def __init__(self, input_size, output_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class AMPTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.place = paddle.CUDAPlace(0)

    def net(self):
        if False:
            print('Hello World!')
        input_size = 4096
        output_size = 4096
        x = static.data(name='X', shape=[1000, 4096], dtype='float32')
        label = static.data(name='Y', shape=[1000, 4096], dtype='float32')
        model = SimpleNet(input_size, output_size)
        mse = paddle.nn.MSELoss()
        out = model(x)
        loss = mse(out, label)
        opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
        opt = paddle.static.amp.decorate(opt, init_loss_scaling=128.0, use_dynamic_loss_scaling=True)
        opt.minimize(loss)
        return (model, loss, opt)

    def test_skip_update(self):
        if False:
            i = 10
            return i + 15
        input_size = 4096
        output_size = 4096
        batch_size = 1000
        nums_batch = 10
        startup_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with static.program_guard(main_prog, startup_prog):
            (model, loss, opt) = self.net()
            weight = model.linear1.weight
            moment1 = opt._optimizer._get_accumulator(opt._optimizer._moment1_acc_str, weight)
            beta_pow1 = opt._optimizer._get_accumulator(opt._optimizer._beta1_pow_acc_str, weight)
            fetch_list = [loss, weight, moment1, beta_pow1, 'find_infinite_scale.tmp_0']
            exe = paddle.static.Executor(self.place)
            train_data = [np.random.rand(batch_size, input_size).astype(np.float32) for _ in range(nums_batch)]
            labels = [np.random.rand(batch_size, output_size).astype(np.float32) for _ in range(nums_batch)]
            (weight_, moment1_, beta_pow1_) = exe.run(startup_prog, fetch_list=[weight, moment1, beta_pow1])
            (pre_weight_, pre_moment1_, pre_beta_pow1_) = (weight_, moment1_, beta_pow1_)
            for i in range(nums_batch):
                if i % 2:
                    train_data[i][10] = np.inf
                (loss_, weight_, moment1_, beta_pow1_, found_inf) = exe.run(main_prog, feed={'X': train_data[i], 'Y': labels[i]}, fetch_list=fetch_list)
                print(loss_, weight_[0][0], moment1_[0][0], beta_pow1_, found_inf)
                if i % 2:
                    self.assertTrue(found_inf)
                    np.testing.assert_array_equal(weight_, pre_weight_)
                    np.testing.assert_array_equal(moment1_, pre_moment1_)
                    np.testing.assert_array_equal(beta_pow1_, pre_beta_pow1_)
                else:
                    self.assertFalse(found_inf)
                    self.assertFalse(np.array_equal(weight_, pre_weight_))
                    self.assertFalse(np.array_equal(moment1_, pre_moment1_))
                    self.assertFalse(np.array_equal(beta_pow1_, pre_beta_pow1_))
                (pre_weight_, pre_moment1_, pre_beta_pow1_) = (weight_, moment1_, beta_pow1_)
if __name__ == '__main__':
    unittest.main()