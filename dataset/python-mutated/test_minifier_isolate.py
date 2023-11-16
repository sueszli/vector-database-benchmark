import functools
import unittest
import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, skipIfRocm, TEST_WITH_ASAN
from torch.utils._triton import has_triton
_HAS_TRITON = has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, 'requires cuda')

class MinifierIsolateTests(MinifierTestBase):

    def _test_after_aot_runtime_error(self, device, expected_error):
        if False:
            while True:
                i = 10
        run_code = f'@torch.compile()\ndef inner(x):\n    x = torch.relu(x)\n    x = torch.cos(x)\n    return x\n\ninner(torch.randn(2, 2).to("{device}"))\n'
        self._run_full_test(run_code, 'aot', expected_error, isolate=True)

    @unittest.skipIf(IS_JETSON, 'Fails on Jetson')
    @inductor_config.patch('cpp.inject_relu_bug_TESTING_ONLY', 'runtime_error')
    def test_after_aot_cpu_runtime_error(self):
        if False:
            while True:
                i = 10
        self._test_after_aot_runtime_error('cpu', '')

    @skipIfRocm
    @requires_cuda()
    @inductor_config.patch('triton.inject_relu_bug_TESTING_ONLY', 'runtime_error')
    def test_after_aot_cuda_runtime_error(self):
        if False:
            i = 10
            return i + 15
        self._test_after_aot_runtime_error('cuda', 'device-side assert')
if __name__ == '__main__':
    import sys
    from torch._dynamo.test_case import run_tests
    if not IS_MACOS and (not TEST_WITH_ASAN) and (sys.version_info < (3, 11)):
        run_tests()