import numpy as np
from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.distributed import fleet
paddle.enable_static()

class TestParallelEmbeddingAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            fleet.init(is_collective=True)
            np.random.seed(2020)
            size = (12, 8)
            np_array = np.random.rand(size[0], size[1])
            paddle.seed(2020)
            data_in = paddle.randint(0, size[0], shape=(10, 4))
            data = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            per_part_size = size[0] // 2
            if rank == 0:
                param_attr = paddle.base.ParamAttr(initializer=paddle.nn.initializer.Assign(np_array[0:per_part_size, :]))
            else:
                param_attr = paddle.base.ParamAttr(initializer=paddle.nn.initializer.Assign(np_array[per_part_size:size[0], :]))
            emb_out = paddle.distributed.split(data_in, size, operation='embedding', num_partitions=2, weight_attr=param_attr)
            return [data_in, emb_out]
if __name__ == '__main__':
    runtime_main(TestParallelEmbeddingAPI, 'parallel_embedding')