import legacy_test.test_collective_api_base as test_base
import paddle
from paddle import base

class TestCollectiveAllgatherObjectAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            object_list = []
            paddle.distributed.all_gather_object(object_list, indata)
            return object_list
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveAllgatherObjectAPI, 'allgather_object')