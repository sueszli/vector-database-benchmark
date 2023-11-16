import os
from test_collective_base_xpu import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestCollectiveAllReduce(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            print('Hello World!')
        ring_id = 0
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            toutdata = main_prog.current_block().create_var(name='outofreduce', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            main_prog.global_block().append_op(type='c_allreduce_sum', inputs={'X': tindata}, attrs={'ring_id': ring_id}, outputs={'Out': toutdata})
            main_prog.global_block().append_op(type='c_sync_comm_stream', inputs={'X': toutdata}, outputs={'Out': toutdata}, attrs={'ring_id': ring_id})
            return toutdata
if __name__ == '__main__':
    os.environ['BKCL_PCIE_RING'] = '1'
    runtime_main(TestCollectiveAllReduce, 'allreduce', 0)