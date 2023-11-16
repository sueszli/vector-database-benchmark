from legacy_test.test_collective_base import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()

class TestCollectiveSendRecvDynamicShape(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            return 10
        ring_id = self.global_ring_id
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype='float64')
            tindata.desc.set_need_check_feed(False)
            if self.rank == 0:
                main_prog.global_block().append_op(type='send_v2', inputs={'X': tindata}, attrs={'ring_id': ring_id, 'peer': 1, 'use_calc_stream': True, 'dynamic_shape': True})
            else:
                main_prog.global_block().append_op(type='recv_v2', outputs={'Out': tindata}, attrs={'peer': 0, 'ring_id': ring_id, 'dtype': tindata.dtype, 'out_shape': tindata.shape, 'use_calc_stream': True, 'dynamic_shape': True})
            return tindata
if __name__ == '__main__':
    runtime_main(TestCollectiveSendRecvDynamicShape, 'sendrecv_dynamic_shape', 0)