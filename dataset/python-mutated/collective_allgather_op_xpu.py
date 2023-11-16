from test_collective_base_xpu import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestCollectiveAllGather(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            i = 10
            return i + 15
        ring_id = 0
        nranks = 2
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[10, 1000], dtype='float32')
            toutdata = main_prog.current_block().create_var(name='outofgather', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            main_prog.global_block().append_op(type='c_allgather', inputs={'X': tindata}, attrs={'ring_id': ring_id, 'nranks': nranks}, outputs={'Out': toutdata})
            main_prog.global_block().append_op(type='c_sync_comm_stream', inputs={'X': toutdata}, outputs={'Out': toutdata}, attrs={'ring_id': ring_id})
            return toutdata
if __name__ == '__main__':
    runtime_main(TestCollectiveAllGather, 'allgather', 0)