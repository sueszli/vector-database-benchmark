import unittest
from dist_pass_test_base import DistPassTestBase
from model_zoo import simple_net
import paddle
from paddle.distributed.passes import PassManager, new_pass

class TestBuildCINNPass(DistPassTestBase):

    def init(self):
        if False:
            i = 10
            return i + 15
        self.atol = 0.0
        self.rtol = 0.0

    def apply_passes(self, main_prog, startup_prog):
        if False:
            for i in range(10):
                print('nop')
        pass_manager = PassManager([new_pass('build_cinn'), new_pass('fuse_elewise_add_act')])
        pass_manager.apply([main_prog], [startup_prog])
        op_types = [op.type for op in main_prog.global_block().ops]
        self.assertTrue('cinn_launch' in op_types)

    def test_bs_32(self):
        if False:
            return 10
        if paddle.is_compiled_with_cinn():
            self.check_main(simple_net, batch_size=32)
if __name__ == '__main__':
    unittest.main()