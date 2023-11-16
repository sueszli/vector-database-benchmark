import legacy_test.test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveGatherAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            while True:
                i = 10
        with base.program_guard(main_prog, startup_program):
            gather_list = []
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                dist.gather(tindata, gather_list, dst=0)
                return [e.cast('float32').numpy() for e in gather_list]
            else:
                tindata = paddle.to_tensor(indata)
                dist.gather(tindata, gather_list, dst=0)
                return [e.numpy() for e in gather_list]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveGatherAPI, 'gather')