from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base, framework
from paddle.base import data_feeder
paddle.enable_static()

def send_new(tensor, dst, group=None, sync_op=True):
    if False:
        while True:
            i = 10
    op_type = 'p_send'
    data_feeder.check_variable_and_dtype(tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], op_type)
    ring_id = 0 if group is None else group.id
    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(type=op_type, inputs={'x': [tensor]}, attrs={'ring_id': ring_id, 'peer': dst, 'dynamic_shape': True})

def recv_new(tensor, src, group=None, sync_op=True, dtype='float32'):
    if False:
        for i in range(10):
            print('nop')
    op_type = 'p_recv'
    data_feeder.check_variable_and_dtype(tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'uint16'], op_type)
    ring_id = 0 if group is None else group.id
    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(type=op_type, outputs={'out': [tensor]}, attrs={'ring_id': ring_id, 'peer': src, 'dynamic_shape': True, 'out_shape': tensor.shape, 'dtype': base.framework.convert_np_dtype_to_dtype_(dtype)})

class TestCollectiveSendRecvAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        if False:
            return 10
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            if rank == 0:
                paddle.distributed.send(tindata, dst=1)
            else:
                paddle.distributed.recv(tindata, src=0)
            return [tindata]

    def get_model_new(self, main_prog, startup_program, rank, dtype='float32', reduce_type=None):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype=dtype)
            if rank == 0:
                send_new(tindata, dst=1)
            else:
                recv_new(tindata, src=0, dtype=dtype)
            return [tindata]
if __name__ == '__main__':
    runtime_main(TestCollectiveSendRecvAPI, 'sendrecv')