import unittest
from dist_pass_test_base import DistPassTestBase
from model_zoo import resnet_model
import paddle
from paddle.distributed.passes import PassManager, new_pass

class TestBuildCINNPass(DistPassTestBase):

    def init(self):
        if False:
            return 10
        self.atol = 0.5
        self.rtol = 0.0

    def apply_passes(self, main_prog, startup_prog):
        if False:
            while True:
                i = 10
        pass_manager = PassManager([new_pass('build_cinn'), new_pass('fuse_elewise_add_act')])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)

    def test_bs_32(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_cinn():
            self.check_main(resnet_model, batch_size=32)
if __name__ == '__main__':
    unittest.main()