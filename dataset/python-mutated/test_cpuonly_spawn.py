import unittest
import paddle
import paddle.distributed as dist
import paddle.optimizer as opt
from paddle import nn

class LinearNet(nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self._linear2(self._linear1(x))

def train(print_result=False):
    if False:
        return 10
    dist.init_parallel_env()
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)
    loss_fn = nn.MSELoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)
    if print_result is True:
        print('loss:', loss.numpy())
    loss.backward()
    print('Grad is', layer._linear1.weight.grad)
    adam.step()
    adam.clear_grad()

class TestSpawn(unittest.TestCase):

    def test_spawn(self):
        if False:
            i = 10
            return i + 15
        dist.spawn(train, backend='gloo', nprocs=4)

    def test_wrong_backend(self):
        if False:
            i = 10
            return i + 15
        try:
            dist.spawn(train, backend='something', nprocs=4)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
if __name__ == '__main__':
    unittest.main()