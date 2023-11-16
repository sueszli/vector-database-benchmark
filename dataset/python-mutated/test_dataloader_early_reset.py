import unittest
import numpy as np
import paddle
from paddle import base

def infinite_reader():
    if False:
        return 10
    num = 0
    while True:
        yield ((np.ones([8, 32]) * num).astype('float32'),)
        num += 1

class TestDataLoaderEarlyReset(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.stop_batch = 10
        self.iterable = True

    def build_network(self):
        if False:
            i = 10
            return i + 15
        y = paddle.static.nn.fc(self.x, size=10)
        loss = paddle.mean(y)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer.minimize(loss)

    def get_place(self):
        if False:
            print('Hello World!')
        if base.is_compiled_with_cuda():
            return base.CUDAPlace(0)
        else:
            return base.CPUPlace()

    def create_data_loader(self):
        if False:
            print('Hello World!')
        self.x = paddle.static.data(name='x', shape=[None, 32], dtype='float32')
        return base.io.DataLoader.from_generator(feed_list=[self.x], capacity=10, iterable=self.iterable)

    def test_main(self):
        if False:
            return 10
        with base.program_guard(base.Program(), base.Program()):
            with base.scope_guard(base.Scope()):
                self.run_network()

    def run_network(self):
        if False:
            print('Hello World!')
        loader = self.create_data_loader()
        self.build_network()
        exe = base.Executor(self.get_place())
        exe.run(base.default_startup_program())
        prog = base.default_main_program()
        loader.set_batch_generator(infinite_reader, places=self.get_place())
        for epoch_id in range(10):
            batch_id = 0
            if loader.iterable:
                for data in loader():
                    (x_val,) = exe.run(prog, feed=data, fetch_list=[self.x])
                    self.assertTrue(np.all(x_val == batch_id))
                    batch_id += 1
                    if batch_id >= self.stop_batch:
                        break
            else:
                loader.start()
                while True:
                    exe.run(prog, fetch_list=[self.x])
                    batch_id += 1
                    if batch_id >= self.stop_batch:
                        loader.reset()
                        break
            self.assertEqual(batch_id, self.stop_batch)
        if loader.iterable:
            loader._reset()

class TestDataLoaderEarlyReset2(TestDataLoaderEarlyReset):

    def setUp(self):
        if False:
            return 10
        self.stop_batch = 20
        self.iterable = False
if __name__ == '__main__':
    unittest.main()