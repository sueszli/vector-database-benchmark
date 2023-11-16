from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base, framework
from paddle.base import data_feeder
paddle.enable_static()

def broadcast_new(tensor, src, group=None, sync_op=True):
    if False:
        for i in range(10):
            print('nop')
    op_type = 'broadcast'
    data_feeder.check_variable_and_dtype(tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], op_type)
    helper = framework.LayerHelper(op_type, **locals())
    ring_id = 0 if group is None else group.id
    helper.append_op(type=op_type, inputs={'x': [tensor]}, outputs={'out': [tensor]}, attrs={'root': src, 'ring_id': ring_id})

class TestCollectiveBroadcastAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype='float32'):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            paddle.distributed.broadcast(tindata, src=1)
            return [tindata]

    def get_model_new(self, main_prog, startup_program, rank, dtype=None, reduce_type=None):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype=dtype)
            tindata.desc.set_need_check_feed(False)
            broadcast_new(tindata, src=1)
            return [tindata]
if __name__ == '__main__':
    runtime_main(TestCollectiveBroadcastAPI, 'broadcast')