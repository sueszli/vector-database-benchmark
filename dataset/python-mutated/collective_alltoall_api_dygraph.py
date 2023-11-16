import legacy_test.test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveAllToAllAPI(test_base.TestCollectiveAPIRunnerBase):

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
            toutdata = []
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                tindata = paddle.split(tindata, 2, axis=0)
                dist.alltoall(tindata, toutdata)
                return [data.cast('float32').numpy() for data in toutdata]
            else:
                tindata = paddle.to_tensor(indata)
                tindata = paddle.split(tindata, 2, axis=0)
                dist.alltoall(tindata, toutdata)
                return [data.numpy() for data in toutdata]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveAllToAllAPI, 'alltoall')