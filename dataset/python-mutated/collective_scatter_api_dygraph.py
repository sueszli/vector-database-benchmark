import sys
sys.path.append('../legacy_test')
import test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveScatterAPI(test_base.TestCollectiveAPIRunnerBase):

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
                (subdata1, subdata2) = paddle.split(tindata, 2, axis=0)
                if rank == 0:
                    dist.scatter(subdata1, src=1)
                else:
                    dist.scatter(subdata1, tensor_list=[subdata1, subdata2], src=1)
                return [subdata1.cast('float32').numpy()]
            else:
                tindata = paddle.to_tensor(indata)
                (subdata1, subdata2) = paddle.split(tindata, 2, axis=0)
                if rank == 0:
                    dist.scatter(subdata1, src=1)
                else:
                    dist.scatter(subdata1, tensor_list=[subdata1, subdata2], src=1)
                return [subdata1.numpy()]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveScatterAPI, 'scatter')