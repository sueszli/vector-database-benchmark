import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
batch = 5
in_dim = 20
out_dim = 10

class cus_tanh(PyLayer):

    @staticmethod
    def forward(ctx, x):
        if False:
            i = 10
            return i + 15
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        if False:
            while True:
                i = 10
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad

class SimpleNet(paddle.nn.Layer):

    def __init__(self, train_id, model_id):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.w = self.create_parameter(shape=[in_dim, batch], dtype='float32')
        self.linear = paddle.nn.Linear(in_dim, out_dim)
        self.tanh = paddle.tanh
        self.trainer_id = train_id
        self.model_id = model_id

    def forward(self, inputs):
        if False:
            return 10
        if self.model_id == 0:
            inputs = cus_tanh.apply(inputs)
        else:
            inputs = self.tanh(inputs)
        inputs = paddle.matmul(self.w, inputs)
        return self.linear(inputs)

class TestDistTraining(unittest.TestCase):

    def test_multiple_gpus(self):
        if False:
            return 10
        self.trainer_id = dist.get_rank()
        dist.init_parallel_env()
        model_a = SimpleNet(self.trainer_id, 0)
        model_b = SimpleNet(self.trainer_id, 1)
        state_dict = model_a.state_dict()
        model_b.set_state_dict(state_dict)
        model_a = paddle.DataParallel(model_a)
        model_b = paddle.DataParallel(model_b)
        for step in range(10):
            x_data = np.random.randn(batch, in_dim).astype(np.float32)
            x = paddle.to_tensor(x_data)
            x.stop_gradient = False
            with model_a.no_sync():
                y_pred_a = model_a(x)
                loss_a = y_pred_a.mean()
                loss_a.backward()
            fused_allreduce_gradients(list(model_a.parameters()), None)
            y_pred_b = model_b(x)
            loss_b = y_pred_b.mean()
            loss_b.backward()
            self.check_gradient(model_a.parameters())
            self.check_gradient(model_b.parameters())
            self.check_acc(model_a._layers.w.grad, model_b._layers.w.grad)
            model_a.clear_gradients()
            model_b.clear_gradients()

    def check_acc(self, grad, acc_grad):
        if False:
            return 10
        grad = grad.numpy(False) if grad is not None else None
        acc_grad = acc_grad.numpy(False) if acc_grad is not None else None
        return np.testing.assert_allclose(grad, acc_grad, rtol=1e-06)

    def broadcast_param(self, param, root):
        if False:
            i = 10
            return i + 15
        paddle.distributed.broadcast(param, root)
        return param

    def check_gradient(self, params):
        if False:
            i = 10
            return i + 15
        other_param = []
        for param in params:
            if param.trainable and param._grad_ivar() is not None:
                grad = param._grad_ivar()
                other_grad = self.broadcast_param(grad.clone(), root=1)
                if self.trainer_id == 0:
                    np.testing.assert_allclose(other_grad.numpy(False), grad.numpy(False))
if __name__ == '__main__':
    unittest.main()