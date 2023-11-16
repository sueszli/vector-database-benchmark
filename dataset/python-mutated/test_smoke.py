import logging
import unittest
import torch
import torch._logging
from torch.testing._internal.common_utils import IS_LINUX, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

class MLP(torch.nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        if False:
            i = 10
            return i + 15
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

def _test_f(x):
    if False:
        while True:
            i = 10
    return x * x

class SmokeTest(TestCase):

    @unittest.skipIf(not HAS_CUDA, 'Triton is not available')
    def test_mlp(self):
        if False:
            while True:
                i = 10
        torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG)
        mlp = torch.compile(MLP().cuda())
        for _ in range(3):
            mlp(torch.randn(1, device='cuda'))
        torch._logging.set_logs()

    @unittest.skipIf(not HAS_CUDA, 'Triton is not available')
    def test_compile_decorator(self):
        if False:
            return 10

        @torch.compile
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sin(x) + x.min()

        @torch.compile(mode='reduce-overhead')
        def bar(x):
            if False:
                return 10
            return x * x
        for _ in range(3):
            foo(torch.full((3, 4), 0.7, device='cuda'))
            bar(torch.rand((2, 2), device='cuda'))

    def test_compile_invalid_options(self):
        if False:
            print('Hello World!')
        with self.assertRaises(RuntimeError):
            opt_f = torch.compile(_test_f, mode='ha')
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if IS_LINUX and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major > 5:
            run_tests()