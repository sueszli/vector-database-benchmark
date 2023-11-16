from legacy_test.test_collective_base import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestCollectiveScatter(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            print('Hello World!')
        ring_id = 0
        rootid = 1
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype='float32')
            tindata.desc.set_need_check_feed(False)
            toutdata = main_prog.current_block().create_var(name='outofreduce', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            main_prog.global_block().append_op(type='c_scatter', inputs={'X': tindata}, attrs={'ring_id': ring_id, 'root': rootid, 'nranks': 2}, outputs={'Out': toutdata})
            main_prog.global_block().append_op(type='c_sync_comm_stream', inputs={'X': toutdata}, outputs={'Out': toutdata}, attrs={'ring_id': ring_id})
            return toutdata
if __name__ == '__main__':
    runtime_main(TestCollectiveScatter, 'scatter', 0)