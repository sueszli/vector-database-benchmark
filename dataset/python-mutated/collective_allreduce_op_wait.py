from legacy_test.test_collective_base import TestCollectiveRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestCollectiveAllreduce(TestCollectiveRunnerBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        if False:
            return 10
        ring_id = 0
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(name='tindata', shape=[-1, 10, 1000], dtype='float32')
            tindata.desc.set_need_check_feed(False)
            toutdata = main_prog.current_block().create_var(name='outofallreduce', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            if True:
                main_prog.global_block().append_op(type='elementwise_add', inputs={'X': tindata, 'Y': tindata}, outputs={'Out': toutdata})
                main_prog.global_block().append_op(type='elementwise_sub', inputs={'X': toutdata, 'Y': tindata}, outputs={'Out': toutdata})
            main_prog.global_block().append_op(type='c_wait_compute', inputs={'X': toutdata}, outputs={'Out': toutdata}, attrs={'ring_id': ring_id})
            main_prog.global_block().append_op(type='c_allreduce_sum', inputs={'X': toutdata}, attrs={'ring_id': ring_id}, outputs={'Out': toutdata}, attr={'use_calc_stream': False})
            main_prog.global_block().append_op(type='c_wait_comm', inputs={'X': toutdata}, outputs={'Out': toutdata}, attrs={'ring_id': ring_id})
            if True:
                main_prog.global_block().append_op(type='elementwise_add', inputs={'X': tindata, 'Y': toutdata}, outputs={'Out': toutdata})
                main_prog.global_block().append_op(type='elementwise_sub', inputs={'X': toutdata, 'Y': tindata}, outputs={'Out': toutdata})
            return toutdata

    def get_model_new_comm(self, main_prog, startup_program, dtype='float32'):
        if False:
            print('Hello World!')
        return self.get_model(main_prog, startup_program)
if __name__ == '__main__':
    runtime_main(TestCollectiveAllreduce, 'allreduce', 0)