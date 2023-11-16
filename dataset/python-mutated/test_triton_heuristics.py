import sys
import unittest
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA
try:
    import triton
except ImportError:
    if __name__ == '__main__':
        sys.exit(0)
    raise unittest.SkipTest('requires triton')
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.triton_heuristics import triton_config

class TestTritonHeuristics(TestCase):

    def test_triton_config(self):
        if False:
            return 10
        '\n        Make sure block size does not exceed the maximum defined in inductor config.\n        '
        cfg = triton_config([2048, 2], 64, 64)
        for label in 'XYZ':
            key = f'{label}BLOCK'
            if key not in cfg.kwargs:
                continue
            self.assertTrue(cfg.kwargs[key] <= config.triton.max_block[label])
if __name__ == '__main__':
    if IS_LINUX and HAS_CUDA:
        run_tests()