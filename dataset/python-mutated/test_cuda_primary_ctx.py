import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocmVersionLessThan, NoTest
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
import sys
import unittest
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest

class TestCudaPrimaryCtx(TestCase):
    CTX_ALREADY_CREATED_ERR_MSG = 'Tests defined in test_cuda_primary_ctx.py must be run in a process where CUDA contexts are never created. Use either run_test.py or add --subprocess to run each test in a different subprocess.'

    @skipIfRocmVersionLessThan((4, 4, 21504))
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        for device in range(torch.cuda.device_count()):
            self.assertFalse(torch._C._cuda_hasPrimaryContext(device), TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG)

    @unittest.skipIf(not TEST_MULTIGPU, 'only one GPU detected')
    def test_str_repr(self):
        if False:
            return 10
        x = torch.randn(1, device='cuda:1')
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        str(x)
        repr(x)
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, 'only one GPU detected')
    def test_copy(self):
        if False:
            print('Hello World!')
        x = torch.randn(1, device='cuda:1')
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        y = torch.randn(1, device='cpu')
        y.copy_(x)
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, 'only one GPU detected')
    def test_pin_memory(self):
        if False:
            while True:
                i = 10
        x = torch.randn(1, device='cuda:1')
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        self.assertFalse(x.is_pinned())
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        x = torch.randn(3, device='cpu').pin_memory()
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        self.assertTrue(x.is_pinned())
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        x = torch.randn(3, device='cpu', pin_memory=True)
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        x = torch.zeros(3, device='cpu', pin_memory=True)
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        x = torch.empty(3, device='cpu', pin_memory=True)
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
        x = x.pin_memory()
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
if __name__ == '__main__':
    run_tests()