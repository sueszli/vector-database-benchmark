import torch
from torch._inductor import metrics
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
aten = torch.ops.aten

def count_bytes_inductor(gm, example_inputs):
    if False:
        for i in range(10):
            print('nop')
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)

def calculate_runtime(f, *args) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Assumes all inputs are fp32\n    '
    metrics.reset()
    torch._dynamo.optimize(count_bytes_inductor)(f)(*args)
    print(metrics.node_runtimes)
    ret = 0.0
    for pair in metrics.node_runtimes:
        ret += pair[1]
    return ret
DEVICE = 'cuda'

def T(*size, dtype=torch.float32, device=DEVICE, grad=False) -> torch.Tensor:
    if False:
        return 10
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)

class TestCase(TorchTestCase):
    device = DEVICE
    "\n    Helper methods to compare runtime estimate against 0. Since this estimate is hardware dependent,\n    stronger comparisons may fail dependending on the host's specs.\n\n    atol/rtol must be provided explicitly with each call, since precision/rel_tol overrides are not always utilized\n    "

    def assertZero(self, x: float):
        if False:
            i = 10
            return i + 15
        assert isinstance(x, float)
        super().assertEqual(x, 0.0, atol=0, rtol=0)

    def assertNotZero(self, x):
        if False:
            return 10
        assert isinstance(x, float)
        super().assertNotEqual(x, 0.0, atol=0, rtol=0)

class UnsupportedTests(TestCase):

    def test_no_op(self):
        if False:
            return 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a
        inp = (T(10, 10),)
        self.assertZero(calculate_runtime(f, *inp))

    def test_no_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                i = 10
                return i + 15
            return a
        inp = (torch.randn((10, 10), device='cpu'),)
        self.assertZero(calculate_runtime(f, *inp))

class ComputeBoundedTests(TestCase):

    def test_conv1d(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return torch.nn.functional.conv1d(x, y)
        inp = (T(33, 16, 30), T(20, 16, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.conv2d(x, y, padding=1)
        inp = (T(8, 4, 3, 3), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d_transpose(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                return 10
            return torch.nn.functional.conv_transpose2d(x, y, padding=1)
        inp = (T(8, 1, 1, 1), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv3d(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.conv3d(x, y)
        inp = (T(20, 16, 50, 10, 20), T(33, 16, 3, 3, 3))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_mm(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return torch.mm(a, b)
        inp = (T(10, 10), T(10, 10))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_addmm(self):
        if False:
            print('Hello World!')

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return torch.addmm(a, b, c)
        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_bmm(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            return torch.bmm(a, b)
        inp = (T(10, 10, 10), T(10, 10, 10))
        self.assertNotZero(calculate_runtime(f, *inp))

class MemoryBoundedTests(TestCase):

    def test_relu(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                i = 10
                return i + 15
            return torch.nn.functional.relu(a)
        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_horizontal_reduction_pointwise(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                i = 10
                return i + 15
            b = a.sum(dim=1)
            c = a.cos()
            return (b, c)
        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_pointwise(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return x.cos()
        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return x.cos()
        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if HAS_CUDA:
        run_tests(needs='filelock')