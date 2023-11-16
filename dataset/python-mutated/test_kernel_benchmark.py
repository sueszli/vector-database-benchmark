import contextlib
import subprocess
import sys
from unittest.mock import patch
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import HAS_CUDA

class TestKernelBenchmark(TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(patch.object(config, 'benchmark_kernel', True))

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.exit_stack.close()

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        PyCodeCache.cache.clear()

    def get_compiled_module(self):
        if False:
            print('Hello World!')
        compiled_module = None
        for v in PyCodeCache.cache.values():
            if hasattr(v, 'benchmark_compiled_module'):
                self.assertTrue(compiled_module is None, 'Found multiple compiled modules')
                compiled_module = v
        self.assertTrue(compiled_module is not None)
        return compiled_module

    def test_kernel_benchmark(self):
        if False:
            return 10

        @torch.compile
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sin(x) + torch.cos(x)
        inp = torch.rand(2, 3).cuda()
        out = f(inp)
        compiled_module = self.get_compiled_module()
        bench_out = subprocess.check_output(f'{sys.executable} {compiled_module.__file__} -kc'.split(), stderr=subprocess.STDOUT).decode()
        FileCheck().check_count('GB/s', 1, exactly=1).run(bench_out)

    def test_bandwidth_computation(self):
        if False:
            print('Hello World!')
        "\n        The test does a matmul and then mul. Without max-autotune, we use\n        the matmul in aten. So there is a single triton kernel for mul.\n        The kernel we generated is like:\n\n            @triton.jit\n            def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):\n\n        Note the in_out_ptr0 argument. It's for a 1000x1000 tensor, but it's\n        inplace udpated, so when computing the bandwidth, we should count\n        the total memory access as 2 * 1000 * 1000 * 4 = 8MB. This amount is\n        what this test asserts.\n        "
        torch.set_float32_matmul_precision('high')

        @torch.compile
        def f(x, y):
            if False:
                print('Hello World!')
            z = x @ y
            w = z * z
            return w
        (M, N, K) = (1000, 1000, 10)
        x = torch.rand(M, K).to('cuda')
        y = torch.rand(K, N).to('cuda')
        out = f(x, y)
        compiled_module = self.get_compiled_module()
        bench_out = subprocess.check_output(f'{sys.executable} {compiled_module.__file__} -k'.split(), stderr=subprocess.STDOUT).decode()
        FileCheck().check_count('0.008 GB ', 1, exactly=1).run(bench_out)
if __name__ == '__main__':
    if HAS_CUDA:
        run_tests()