import subprocess
import sys
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.codecache import PyCodeCache
from torch.testing._internal.inductor_utils import HAS_CUDA

class TestTritonWrapper(TestCase):

    def get_compiled_module(self):
        if False:
            for i in range(10):
                print('nop')
        compiled_module = None
        for v in PyCodeCache.cache.values():
            if hasattr(v, 'benchmark_compiled_module'):
                self.assertTrue(compiled_module is None, 'Found multiple compiled modules')
                compiled_module = v
        self.assertTrue(compiled_module is not None)
        return compiled_module

    def test_wrapper_using_cuda_seed(self):
        if False:
            i = 10
            return i + 15
        '\n        Make sure the subprocess.check_output does not throw.\n        '

        @torch.compile
        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            z = torch.nn.functional.dropout(x, 0.5)
            return z + y
        N = 10
        x = torch.rand(N).to('cuda')
        y = torch.rand(N).to('cuda')
        out = f(x, y)
        compiled_module = self.get_compiled_module()
        bench_out = subprocess.check_output(f'{sys.executable} {compiled_module.__file__}'.split(), stderr=subprocess.STDOUT).decode()
        self.assertTrue(len(bench_out) > 0)
if __name__ == '__main__':
    if HAS_CUDA:
        run_tests()