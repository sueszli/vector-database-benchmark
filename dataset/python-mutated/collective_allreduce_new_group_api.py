from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()

class TestCollectiveAllreduceNewGroupAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            while True:
                i = 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[1, 10, 1000], dtype='float32')
            gp = paddle.distributed.new_group([0, 1])
            paddle.distributed.all_reduce(tindata, group=gp, sync_op=True)
            return [tindata]
if __name__ == '__main__':
    runtime_main(TestCollectiveAllreduceNewGroupAPI, 'allreduce')