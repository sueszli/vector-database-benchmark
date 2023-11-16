import test_collective_api_base as test_base
import paddle.distributed as dist
from paddle import base

class TestCollectiveBroadcastObjectListAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            print('Hello World!')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            object_list = [indata]
            dist.broadcast_object_list(object_list, src=1)
            return object_list
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveBroadcastObjectListAPI, 'broadcast_object_list')