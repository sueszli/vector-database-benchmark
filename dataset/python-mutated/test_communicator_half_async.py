import os
import subprocess
import sys
import unittest
import numpy
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()

class TestCommunicatorHalfAsyncEnd2End(unittest.TestCase):

    def net(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
        y_predict = paddle.static.nn.fc(x, size=1, activation=None)
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)
        return (avg_cost, x, y)

    def fake_reader(self):
        if False:
            print('Hello World!')

        def reader():
            if False:
                return 10
            for i in range(10000):
                x = numpy.random.random((1, 13)).astype('float32')
                y = numpy.random.randint(0, 2, (1, 1)).astype('int64')
                yield (x, y)
        return reader

    def run_pserver(self, role, strategy):
        if False:
            while True:
                i = 10
        fleet.init(role)
        (avg_cost, x, y) = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
        fleet.init_server()
        fleet.run_server()

    def run_trainer(self, role, strategy):
        if False:
            print('Hello World!')
        place = base.core.CPUPlace()
        exe = base.Executor(place)
        fleet.init(role)
        (avg_cost, x, y) = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()
        train_reader = paddle.batch(self.fake_reader(), batch_size=24)
        feeder = base.DataFeeder(place=place, feed_list=[x, y])
        for (batch_id, data) in enumerate(train_reader()):
            exe.run(paddle.static.default_main_program(), feed=feeder.feed(data), fetch_list=[])
        fleet.stop_worker()

    def run_ut(self):
        if False:
            print('Hello World!')
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        training_role = os.getenv('TRAINING_ROLE', 'TRAINER')
        role = role_maker.UserDefinedRoleMaker(current_id=0, role=role_maker.Role.WORKER if training_role == 'TRAINER' else role_maker.Role.SERVER, worker_num=1, server_endpoints=['127.0.0.1:6002'])
        if training_role == 'TRAINER':
            self.run_trainer(role, strategy)
        else:
            self.run_pserver(role, strategy)

    def test_communicator(self):
        if False:
            print('Hello World!')
        run_server_cmd = '\n\nimport sys\nimport os\n\nimport time\nimport threading\nimport subprocess\nimport unittest\nimport numpy\n\nfrom test_communicator_half_async import TestCommunicatorHalfAsyncEnd2End\n\nimport paddle\nimport paddle.base as base\nimport paddle.distributed.fleet as fleet\nimport paddle.distributed.fleet.base.role_maker as role_maker\n\npaddle.enable_static()\n\nclass RunServer(TestCommunicatorHalfAsyncEnd2End):\n    def runTest(self):\n        pass\n\nos.environ["http_proxy"] = ""\nos.environ["https_proxy"] = ""\nos.environ["TRAINING_ROLE"] = "PSERVER"\nhalf_run_server = RunServer()\nhalf_run_server.run_ut()\n'
        server_file = 'run_server_for_communicator_haflaysnc.py'
        with open(server_file, 'w') as wb:
            wb.write(run_server_cmd)
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        _python = sys.executable
        ps_cmd = f'{_python} {server_file}'
        ps_proc = subprocess.Popen(ps_cmd.strip().split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['FLAGS_communicator_send_queue_size'] = '1'
        os.environ['FLAGS_communicator_max_merge_var_num'] = '1'
        self.run_ut()
        ps_proc.kill()
        if os.path.exists(server_file):
            os.remove(server_file)
if __name__ == '__main__':
    unittest.main()