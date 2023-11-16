import test_collective_api_base as test_base
import paddle
import paddle.distributed as dist
from paddle import base

class TestCollectiveAllgatherAPI(test_base.TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            print('Hello World!')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            while True:
                i = 10
        with base.program_guard(main_prog, startup_program):
            tensor_list = []
            if indata.dtype == 'bfloat16':
                tindata = paddle.to_tensor(indata, 'float32').cast('uint16')
                dist.all_gather(tensor_list, tindata)
                return [tensor.cast('float32').numpy() for tensor in tensor_list]
            else:
                tindata = paddle.to_tensor(indata)
                dist.all_gather(tensor_list, tindata)
                return [tensor.numpy() for tensor in tensor_list]
if __name__ == '__main__':
    test_base.runtime_main(TestCollectiveAllgatherAPI, 'allgather')