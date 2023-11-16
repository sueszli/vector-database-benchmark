from test_collective_base_xpu import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.base import core, layers
paddle.enable_static()

class TestCollectiveConcat(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            print('Hello World!')
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            print('Hello World!')
        ring_id = 0
        nranks = 2
        with base.program_guard(main_prog, startup_program):
            tindata = layers.data(name='tindata', shape=[10, 1000], dtype='float32')
            toutdata = main_prog.current_block().create_var(name='outofconcat', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            main_prog.global_block().append_op(type='c_concat', inputs={'X': tindata}, attrs={'ring_id': ring_id, 'rank': self.rank, 'nranks': nranks}, outputs={'Out': toutdata})
            return toutdata
if __name__ == '__main__':
    runtime_main(TestCollectiveConcat, 'c_concat', 0)