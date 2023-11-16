from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
import paddle.distributed as dist
from paddle import base, framework
from paddle.base import data_feeder
paddle.enable_static()

def all_reduce_new(tensor, reduce_type=str(dist.ReduceOp.SUM), group=None):
    if False:
        print('Hello World!')
    op_type = 'all_reduce'
    data_feeder.check_variable_and_dtype(tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], op_type)
    ring_id = 0 if group is None else group.id
    if not isinstance(ring_id, int):
        raise ValueError("The type of 'ring_id' for all_reduce should be int.")
    helper = framework.LayerHelper(op_type, **locals())
    if not reduce_type.isdigit():
        raise ValueError("The type of 'reduce_type' for all_reduce should be int.")
    helper.append_op(type=op_type, inputs={'x': [tensor]}, outputs={'out': [tensor]}, attrs={'ring_id': ring_id, 'reduce_type': int(reduce_type)})

class TestCollectiveAllreduceAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            paddle.distributed.all_reduce(tindata)
            return [tindata]

    def get_model_new(self, main_prog, startup_program, rank, dtype='float32', reduce_type=str(dist.ReduceOp.SUM)):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            all_reduce_new(tindata, reduce_type)
            return [tindata]

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            print('Hello World!')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            paddle.distributed.all_reduce(tindata)
            return [tindata]
if __name__ == '__main__':
    runtime_main(TestCollectiveAllreduceAPI, 'allreduce')