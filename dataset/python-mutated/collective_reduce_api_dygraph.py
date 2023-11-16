import sys
sys.path.append('../legacy_test')
import test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveReduceAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                dist.reduce(tindata, dst=0)
                return [tindata.cast('float32').numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                dist.reduce(tindata, dst=0)
                return [tindata.numpy()]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveReduceAPI, 'reduce')