from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()

class TestCollectiveReduceScatterAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_model_new(self, main_prog, startup_program, rank, dtype='float32', reduce_type=None):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            toutdata = paddle.static.data(name='toutdata', shape=[5, 1000], dtype=dtype)
            paddle.distributed.reduce_scatter(toutdata, tindata)
            return [toutdata]

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            toutdata = paddle.static.data(name='toutdata', shape=[5, 1000], dtype=dtype)
            paddle.distributed.reduce_scatter(toutdata, tindata)
            return [toutdata]
if __name__ == '__main__':
    runtime_main(TestCollectiveReduceScatterAPI, 'reduce_scatter')