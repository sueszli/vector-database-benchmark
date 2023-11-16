import os
import unittest
import paddle
from paddle import base
from paddle.distributed import fleet
paddle.enable_static()

class TestFleetExecutor(unittest.TestCase):

    def run_fleet_executor(self, place, fleet_opt={}):
        if False:
            while True:
                i = 10
        exe = paddle.static.Executor(place)
        empty_program = paddle.static.Program()
        with base.program_guard(empty_program, empty_program):
            x = paddle.static.data(name='x', shape=[-1, 1], dtype=paddle.float32)
        empty_program._pipeline_opt = {'fleet_opt': fleet_opt, 'section_program': empty_program}
        exe.run(empty_program, feed={'x': [1]})

    def test_dist_executor_on_multi_devices(self):
        if False:
            print('Hello World!')
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002,127.0.0.1:7003,127.0.0.1:7004,127.0.0.1:7005,127.0.0.1:7006,127.0.0.1:7007'
        strategy = fleet.DistributedStrategy()
        strategy.sharding_configs = {'dp_degree': 2, 'mp_degree': 2, 'pp_degree': 2}
        strategy.pipeline_configs = {'accumulate_steps': 8}
        fleet_opt = {'dist_strategy': strategy.sharding_configs, 'num_micro_batches': strategy.pipeline_configs['accumulate_steps']}
        if base.is_compiled_with_cuda():
            pass
if __name__ == '__main__':
    unittest.main()