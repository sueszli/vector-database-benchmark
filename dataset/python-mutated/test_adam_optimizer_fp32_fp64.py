import unittest
import paddle
from paddle import base

def get_places():
    if False:
        print('Hello World!')
    places = [base.CPUPlace()]
    if base.is_compiled_with_cuda():
        places.append(base.CUDAPlace(0))
    return places

def main_test_func(place, dtype):
    if False:
        while True:
            i = 10
    main = base.Program()
    startup = base.Program()
    with base.program_guard(main, startup):
        with base.scope_guard(base.Scope()):
            x = paddle.static.data(name='x', shape=[None, 13], dtype=dtype)
            y = paddle.static.data(name='y', shape=[None, 1], dtype=dtype)
            y_predict = paddle.static.nn.fc(x, size=1)
            cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
            avg_cost = paddle.mean(cost)
            adam_optimizer = paddle.optimizer.Adam(0.01)
            adam_optimizer.minimize(avg_cost)
            fetch_list = [avg_cost]
            train_reader = paddle.batch(paddle.dataset.uci_housing.train(), batch_size=1)
            feeder = base.DataFeeder(place=place, feed_list=[x, y])
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

class AdamFp32Test(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float32'

    def test_main(self):
        if False:
            return 10
        for p in get_places():
            main_test_func(p, self.dtype)

class AdamFp64Test(AdamFp32Test):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dtype = 'float64'
if __name__ == '__main__':
    unittest.main()