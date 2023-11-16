import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.nn import Linear
paddle.seed(1024)
np.random.seed(2021)
batch = 5
in_dim = 10
out_dim = 20

class SimpleNet(paddle.nn.Layer):

    def __init__(self, train_id):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.w1 = self.create_parameter(shape=[in_dim, out_dim], dtype='float32')
        self.w2 = self.create_parameter(shape=[in_dim, out_dim], dtype='float32')
        self.share_net = Linear(out_dim, 10)
        self.unused_param = self.create_parameter(shape=[out_dim, in_dim], dtype='float64')
        self.trainer_id = train_id

    def forward(self, x):
        if False:
            return 10
        is_use = paddle.equal_all(x, paddle.ones(shape=(batch, in_dim))).item() and self.trainer_id == 1
        if is_use:
            tmp = paddle.matmul(x, self.w1)
        else:
            tmp = paddle.matmul(x, self.w2)
        return self.share_net(tmp)

class TestDistTraining(unittest.TestCase):

    def test_multiple_gpus(self):
        if False:
            print('Hello World!')
        self.trainer_id = dist.get_rank()
        self.pg = dist.init_parallel_env()
        model_a = SimpleNet(self.trainer_id)
        model_b = SimpleNet(self.trainer_id)
        state_dict = model_a.state_dict()
        model_b.set_state_dict(state_dict)
        model_a = paddle.DataParallel(model_a, find_unused_parameters=True, group=self.pg)
        model_b = paddle.DataParallel(model_b, find_unused_parameters=True, group=self.pg)
        ones_input = paddle.ones(shape=(batch, in_dim))
        ones_input.stop_gradient = True
        w1_grad_sum = np.zeros((in_dim, out_dim), dtype='float32')
        w2_grad_sum = np.zeros((in_dim, out_dim), dtype='float32')
        for step_id in range(5):
            random_input = paddle.rand(shape=(batch, in_dim))
            random_input.stop_gradient = True
            if step_id % 2 == 0:
                out_a = model_a(random_input)
                out_b = model_b(random_input)
            else:
                out_a = model_a(ones_input)
                out_b = model_b(ones_input)
            out_a.sum().backward()
            out_b.sum().backward()
            self.check_gradient(model_a.parameters())
            self.check_gradient(model_b.parameters())
            w1_grad_sum = self.check_acc(model_a._layers.w1.grad, w1_grad_sum, model_b._layers.w1.grad)
            w2_grad_sum = self.check_acc(model_a._layers.w2.grad, w2_grad_sum, model_b._layers.w2.grad)
            model_a.clear_gradients()

    def check_acc(self, grad, grad_sum, acc_grad):
        if False:
            for i in range(10):
                print('nop')
        if grad is not None:
            grad_sum = grad_sum + grad.numpy(False)
            acc_grad = acc_grad.numpy(False) if acc_grad is not None else None
            np.testing.assert_allclose(grad_sum, acc_grad, rtol=1e-06)
        return grad_sum

    def print_trainer_0(self, *args):
        if False:
            while True:
                i = 10
        if self.trainer_id == 0:
            print(*args)

    def broadcast_param(self, param, root):
        if False:
            i = 10
            return i + 15
        self.pg.process_group.broadcast(param, root)
        return param

    def check_gradient(self, params):
        if False:
            i = 10
            return i + 15
        other_param = []
        for param in params:
            if param.trainable and param.grad is not None:
                grad = param.grad
                other_grad = self.broadcast_param(grad, root=1)
                if self.trainer_id == 0:
                    np.testing.assert_allclose(other_grad.numpy(False), grad.numpy(False))
if __name__ == '__main__':
    unittest.main()