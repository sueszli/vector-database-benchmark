import legacy_test.test_collective_api_base as test_base
import paddle.distributed as dist
from paddle import base

class TestCollectiveScatterObjectListAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            data_len = len(indata) // 2
            in_object_list = [indata[:data_len], indata[data_len:]]
            out_object_list = []
            dist.scatter_object_list(out_object_list, in_object_list, src=1)
            return out_object_list
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveScatterObjectListAPI, 'scatter_object_list')