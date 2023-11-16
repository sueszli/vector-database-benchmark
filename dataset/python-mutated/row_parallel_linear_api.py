import numpy as np
from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.distributed import fleet
paddle.enable_static()

class TestRowParallelLinearAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            fleet.init(is_collective=True)
            np.random.seed(2020)
            np_array = np.random.rand(1000, 16)
            data = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            paddle.distributed.broadcast(data, src=0)
            data = paddle.split(data, 2, axis=1)[rank]
            if rank == 0:
                param_attr = paddle.base.ParamAttr(initializer=paddle.nn.initializer.Assign(np_array[0:500, :]))
            else:
                param_attr = paddle.base.ParamAttr(initializer=paddle.nn.initializer.Assign(np_array[500:1000, :]))
            linear_out = paddle.distributed.split(data, size=(1000, 16), operation='linear', axis=0, num_partitions=2, weight_attr=param_attr, bias_attr=True)
            return [linear_out]
if __name__ == '__main__':
    runtime_main(TestRowParallelLinearAPI, 'row_parallel_linear')