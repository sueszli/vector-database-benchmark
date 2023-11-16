import unittest
from test_buffer_shared_memory_reuse_pass import InplaceTestBase

class CUDAInplaceTestWithFuseOptimizationOps(InplaceTestBase):

    def initParameter(self):
        if False:
            while True:
                i = 10
        self.use_cuda = True
        self.fuse_all_optimizer_ops = True
        self.fuse_all_reduce_ops = False

    def test_single_card_fetch_var(self):
        if False:
            i = 10
            return i + 15
        self.check_single_card_fetch_var()

class CPUInplaceTestWithFuseOptimizationOps(InplaceTestBase):

    def initParameter(self):
        if False:
            print('Hello World!')
        self.use_cuda = False
        self.fuse_all_optimizer_ops = True
        self.fuse_all_reduce_ops = False

    @unittest.skip('should fix this later.')
    def test_single_card_fetch_var(self):
        if False:
            print('Hello World!')
        self.check_single_card_fetch_var()
if __name__ == '__main__':
    unittest.main()