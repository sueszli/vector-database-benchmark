import legacy_test.test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveAllToAllSingleAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                toutdata = paddle.to_tensor(tindata, 'float32').cast('uint16')
                dist.alltoall_single(tindata, toutdata)
                return [toutdata.cast('float32').numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                toutdata = paddle.to_tensor(indata)
                dist.alltoall_single(tindata, toutdata)
                return [toutdata.numpy()]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveAllToAllSingleAPI, 'alltoall')