import test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveAllreduceAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                dist.all_reduce(tindata)
                return [tindata.cast('float32').numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                dist.all_reduce(tindata)
                return [tindata.numpy()]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveAllreduceAPI, 'allreduce')