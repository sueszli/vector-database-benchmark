import sys
import numpy as np
sys.path.append('..')
from legacy_test.test_dist_base import TestParallelDyGraphRunnerBase, runtime_main
import paddle
np.random.seed(2021)
paddle.seed(1024)
batch_size = 4
batch_num = 1000

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.net_a = paddle.nn.Sequential(paddle.nn.Linear(10, 20), paddle.nn.Linear(20, 20), paddle.nn.Linear(20, 5))
        self.net_b = paddle.nn.Sequential(paddle.nn.Linear(10, 20), paddle.nn.Linear(20, 20), paddle.nn.Linear(20, 5))
        self.step = 0

    def forward(self, x):
        if False:
            print('Hello World!')
        return paddle.to_tensor([0.0], dtype='float32')

def fake_sample_reader():
    if False:
        i = 10
        return i + 15

    def __reader__():
        if False:
            i = 10
            return i + 15
        for i in range(batch_num):
            x_data = np.random.random_sample((10,)).astype('float32')
            yield x_data
    return __reader__

class TestSimpleNet(TestParallelDyGraphRunnerBase):

    def get_model(self):
        if False:
            return 10
        model = SimpleNet()
        train_reader = paddle.batch(fake_sample_reader(), batch_size=batch_size, drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
        return (model, train_reader, optimizer)

    def run_one_loop(self, model, optimizer, batch):
        if False:
            while True:
                i = 10
        x_data = np.array(list(batch))
        x_data = x_data.reshape((-1, 10))
        x = paddle.to_tensor(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss
if __name__ == '__main__':
    runtime_main(TestSimpleNet)