import unittest
import numpy as np
import paddle
from paddle import base, nn
from paddle.distributed import fleet
paddle.enable_static()
fleet.init(is_collective=True)

class SimpleNet(nn.Layer):

    def __init__(self, input_size, output_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class TestFleetWithQAT(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.input_size = 4096
        self.output_size = 4096
        self.batch_size = 8

    def setup_strategy(self, strategy):
        if False:
            i = 10
            return i + 15
        strategy.qat = True

    def generate_program(self, strategy):
        if False:
            print('Hello World!')
        (train_prog, startup_prog) = (base.Program(), base.Program())
        with base.program_guard(train_prog, startup_prog):
            input_x = paddle.static.data(name='X', shape=[self.batch_size, self.input_size], dtype='float32')
            input_y = paddle.static.data(name='Y', shape=[self.batch_size, self.output_size], dtype='float32')
            model = SimpleNet(self.input_size, self.output_size)
            mse = paddle.nn.MSELoss()
            out = model(input_x)
            loss = mse(out, input_y)
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            optimizer.minimize(loss)
        return (train_prog, startup_prog, input_x, input_y, optimizer)

    def execute_program(self, train_prog, startup_prog, input_x, input_y):
        if False:
            print('Hello World!')
        place = base.CUDAPlace(0) if paddle.base.is_compiled_with_cuda() else base.CPUPlace()
        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)
        data = (np.random.randn(self.batch_size, self.input_size), np.random.randn(self.batch_size, self.output_size))
        exe.run(train_prog, feed=feeder.feed([data]))

    def valid_program(self, train_prog, eval_prog):
        if False:
            print('Hello World!')
        ops_type = [op.type for op in train_prog.block(0).ops]
        self.assertEqual(ops_type.count('matmul_v2'), 3)
        self.assertEqual(ops_type.count('quantize_linear'), 6)
        self.assertEqual(ops_type.count('dequantize_linear'), 6)

    def test_fleet_with_qat(self):
        if False:
            print('Hello World!')
        dist_strategy = paddle.distributed.fleet.DistributedStrategy()
        self.setup_strategy(dist_strategy)
        (train_prog, startup_prog, input_x, input_y, optimizer) = self.generate_program(dist_strategy)
        place = base.CUDAPlace(0) if paddle.base.is_compiled_with_cuda() else base.CPUPlace()
        eval_prog = train_prog.clone(for_test=True)
        optimizer.qat_init(place, scope=paddle.static.global_scope(), test_program=eval_prog)
        self.execute_program(train_prog, startup_prog, input_x, input_y)
        self.valid_program(train_prog, eval_prog)

class TestFleetWithAMPQAT(TestFleetWithQAT):

    def setup_strategy(self, strategy):
        if False:
            return 10
        strategy.qat = True
        strategy.amp = True

    def valid_program(self, train_prog, eval_prog):
        if False:
            return 10
        ops_type = [op.type for op in train_prog.block(0).ops]
        self.assertEqual(ops_type.count('matmul_v2'), 3)
        self.assertEqual(ops_type.count('quantize_linear'), 6)
        self.assertEqual(ops_type.count('dequantize_linear'), 6)
if __name__ == '__main__':
    unittest.main()