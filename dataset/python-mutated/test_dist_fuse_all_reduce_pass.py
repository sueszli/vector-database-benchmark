import unittest
from dist_pass_test_base import DistPassTestBase
from model_zoo import resnet_model
from paddle.distributed.passes import PassManager, new_pass

class TestFuseAllReducePass(DistPassTestBase):

    def init(self):
        if False:
            print('Hello World!')
        self.atol = 0.0
        self.rtol = 0.0

    def apply_passes(self, main_prog, startup_prog):
        if False:
            for i in range(10):
                print('nop')
        pass_manager = PassManager([new_pass('fuse_elewise_add_act'), new_pass('fuse_all_reduce', {'max_memory_size': 1024 * 1024})])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)

    def test_bs_32(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_main(resnet_model, batch_size=32)
if __name__ == '__main__':
    unittest.main()