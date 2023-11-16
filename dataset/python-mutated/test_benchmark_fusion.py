import math
import os
import sys
import torch
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, skipIfRocm, TEST_WITH_ASAN, TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import contextlib
import unittest
from torch._inductor import config
from torch._inductor.scheduler import Scheduler
if IS_WINDOWS and IS_CI:
    sys.stderr.write('Windows CI does not have necessary dependencies for test_torchinductor yet\n')
    if __name__ == '__main__':
        sys.exit(0)
    raise unittest.SkipTest('requires sympy/functorch/filelock')
from inductor.test_torchinductor import check_model, check_model_cuda, copy_tests

class TestCase(TorchTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(config.patch({'benchmark_kernel': True, 'benchmark_fusion': True}))

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls._stack.close()
        super().tearDownClass()

class BenchmarkFusionTestTemplate:

    def test_softmax(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            return torch.nn.functional.softmax(x, dim=-1)
        self.common(f, (torch.rand(2, 8192),))

    @skipIfRocm
    def test_resnet18(self):
        if False:
            i = 10
            return i + 15
        import torchvision
        model = torchvision.models.resnet18()
        model.eval()
        batch_size = 16
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
        self.common(model, inputs, atol=0.01, rtol=0.01)

    def test_register_spills(self):
        if False:
            print('Hello World!')
        '\n        The test can potentially trigger register spills\n        '
        old_benchmark_fn = Scheduler.benchmark_fused_nodes

        def new_benchmark_fn(scheduler, nodes):
            if False:
                return 10
            '\n            We override Scheduler.benchmark_fused_nodes to return latency 1.0\n            if there are no register spills. Without this, we may not able to\n            test the code path handling register spilling because before register\n            start spilling, the related fusion may have already been skipped\n            due to longer lantency.\n            '
            (ms, path) = old_benchmark_fn(scheduler, nodes)
            if not math.isinf(ms):
                ms = 1.0
            return (ms, path)
        with unittest.mock.patch.object(Scheduler, 'benchmark_fused_nodes', new_benchmark_fn), config.patch('dynamic_scale_rblock', False):
            S = 512

            def f(*inputs):
                if False:
                    i = 10
                    return i + 15
                inputs = list(inputs)
                outputs = []
                out = torch.zeros(S, device=self.device)
                for x in inputs:
                    x = x * 2
                    x = x + 1
                    x = x.sum(dim=-1)
                    outputs.append(x)
                    out = out + x
                return (outputs, out)
            N = int(os.environ.get('NINP', '30'))
            inputs = [torch.randn(S, 2560, device=self.device) for _ in range(N)]
            opt_f = torch.compile(f)
            opt_f(*inputs)
if HAS_CUDA and (not TEST_WITH_ASAN):

    class BenchmarkFusionCudaTest(TestCase):
        common = check_model_cuda
        device = 'cuda'
    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCudaTest, 'cuda')
if HAS_CPU and (not torch.backends.mps.is_available()):

    class BenchmarkFusionCpuTest(TestCase):
        common = check_model
        device = 'cpu'
    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCpuTest, 'cpu')
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if HAS_CPU or HAS_CUDA:
        run_tests()