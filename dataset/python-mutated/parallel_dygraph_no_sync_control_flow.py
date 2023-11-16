import numpy as np
from legacy_test.test_dist_base import runtime_main
from parallel_dygraph_no_sync import TestNoSync
import paddle
from paddle.nn import Linear
seed = 90
RUN_STEP = 20
batch_size = 4
batch_num = 1000

class SimpleNetControlFlow(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.net_a = Linear(10, 20)
        self.net_b = Linear(20, 5)
        self.net_c = Linear(5, 10)
        self.step = 0

    def forward(self, x):
        if False:
            print('Hello World!')
        self.step = self.step + 1
        x = self.net_a(x)
        if self.step > 10:
            x.stop_gradient = True
        x = self.net_b(x)
        x = self.net_c(x)
        return x

class TestNoSyncControlFlow(TestNoSync):

    def get_model(self):
        if False:
            print('Hello World!')
        model = SimpleNetControlFlow()
        train_reader = paddle.batch(fake_sample_reader(), batch_size=batch_size, drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
        return (model, train_reader, optimizer)

    def run_one_loop(self, model, optimizer, batch):
        if False:
            return 10
        x_data = np.array(list(batch))
        x_data = x_data.reshape((-1, 10))
        x = paddle.to_tensor(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss

def fake_sample_reader():
    if False:
        for i in range(10):
            print('nop')

    def __reader__():
        if False:
            while True:
                i = 10
        for i in range(batch_num):
            x_data = np.random.random_sample((10,)).astype('float32')
            yield x_data
    return __reader__
if __name__ == '__main__':
    runtime_main(TestNoSyncControlFlow)