import sys
sys.path.append('../legacy_test')
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()

class TestCollectiveScatterAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            print('Hello World!')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            toutdata = paddle.tensor.fill_constant(shape=[5, 1000], dtype='float32', value=1.0)
            tensor_list = None
            if rank == 1:
                tensor_list = paddle.split(tindata, 2, axis=0)
            paddle.distributed.scatter(toutdata, tensor_list, src=1)
            return [toutdata]

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            while True:
                i = 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            toutdata = paddle.tensor.fill_constant(shape=[5, 1000], dtype=dtype, value=1.0)
            tensor_list = None
            if rank == 1:
                tensor_list = paddle.split(tindata, 2, axis=0)
            paddle.distributed.scatter(toutdata, tensor_list, src=1)
            return [toutdata]
if __name__ == '__main__':
    runtime_main(TestCollectiveScatterAPI, 'scatter')