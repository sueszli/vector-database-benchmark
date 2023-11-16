import sys
sys.path.append('../legacy_test')
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()

class TestCollectiveBarrierAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            while True:
                i = 10
        with base.program_guard(main_prog, startup_program):
            paddle.distributed.barrier()
            return []

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            paddle.distributed.barrier()
            return []
if __name__ == '__main__':
    runtime_main(TestCollectiveBarrierAPI, 'barrier')