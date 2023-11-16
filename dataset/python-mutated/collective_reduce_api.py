import sys
sys.path.append('../legacy_test')
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
import paddle.distributed as dist
from paddle import base, framework
from paddle.base import data_feeder
paddle.enable_static()

def reduce_new(tensor, dst, reduce_type=str(dist.ReduceOp.SUM), group=None):
    if False:
        while True:
            i = 10
    op_type = 'reduce'
    data_feeder.check_variable_and_dtype(tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], op_type)
    ring_id = 0 if group is None else group.id
    helper = framework.LayerHelper(op_type, **locals())
    if not reduce_type.isdigit():
        raise ValueError("The type of 'reduce_type' for reduce should be int.")
    helper.append_op(type=op_type, inputs={'x': [tensor]}, outputs={'out': [tensor]}, attrs={'ring_id': ring_id, 'root_id': dst, 'reduce_type': int(reduce_type)})

class TestCollectiveReduceAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype='float32')
            tindata.desc.set_need_check_feed(False)
            paddle.distributed.reduce(tindata, dst=0)
            return [tindata]

    def get_model_new(self, main_prog, startup_program, rank, dtype='float32', reduce_type=str(dist.ReduceOp.SUM)):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            reduce_new(tindata, dst=0, reduce_type=reduce_type)
            return [tindata]

    def get_model_new_comm(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            paddle.distributed.reduce(tindata, dst=0)
            return [tindata]
if __name__ == '__main__':
    runtime_main(TestCollectiveReduceAPI, 'reduce')