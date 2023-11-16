import legacy_test.test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveReduceScatterAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                (subdata1, subdata2) = paddle.split(tindata, 2, axis=0)
                dist.reduce_scatter(subdata1, [subdata1, subdata2])
                return [subdata1.cast('float32').numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                (subdata1, subdata2) = paddle.split(tindata, 2, axis=0)
                dist.reduce_scatter(subdata1, [subdata1, subdata2])
                return [subdata1.numpy()]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveReduceScatterAPI, 'reduce_scatter')