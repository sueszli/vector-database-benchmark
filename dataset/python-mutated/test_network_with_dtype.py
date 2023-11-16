import unittest
import paddle
from paddle import base
from paddle.base import core
BATCH_SIZE = 20

class TestNetWithDtype(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float64'
        self.init_dtype()

    def run_net_on_place(self, place):
        if False:
            for i in range(10):
                print('nop')
        main = base.Program()
        startup = base.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[-1, 13], dtype=self.dtype)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype=self.dtype)
            y_predict = paddle.static.nn.fc(x, size=1, activation=None)
            cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
            avg_cost = paddle.mean(cost)
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)
        fetch_list = [avg_cost]
        train_reader = paddle.batch(paddle.dataset.uci_housing.train(), batch_size=BATCH_SIZE)
        feeder = base.DataFeeder(place=place, feed_list=[x, y])
        exe = base.Executor(place)
        exe.run(startup)
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
            break

    def init_dtype(self):
        if False:
            print('Hello World!')
        pass

    def test_cpu(self):
        if False:
            return 10
        place = base.CPUPlace()
        self.run_net_on_place(place)

    def test_gpu(self):
        if False:
            while True:
                i = 10
        if not core.is_compiled_with_cuda():
            return
        place = base.CUDAPlace(0)
        self.run_net_on_place(place)
if __name__ == '__main__':
    unittest.main()