import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.pipeline.sync import Pipe
from torch.testing._internal.common_utils import run_tests

def test_python_autograd_function(setup_rpc):
    if False:
        print('Hello World!')

    class Identity(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if False:
                i = 10
                return i + 15
            return input

        @staticmethod
        def backward(ctx, grad):
            if False:
                print('Hello World!')
            return grad

    class M(nn.Module):

        def forward(self, input):
            if False:
                i = 10
                return i + 15
            return Identity.apply(input)
    model = nn.Sequential(M(), M())
    model = Pipe(model, checkpoint='always')
    x = torch.rand(42)
    y = model(x)
    assert torch.allclose(x, y.local_value())

def test_exception_no_hang(setup_rpc):
    if False:
        for i in range(10):
            print('nop')

    class ExpectedException(Exception):
        pass

    class Pass(nn.Module):

        def forward(self, x):
            if False:
                i = 10
                return i + 15
            return x

    class Raise(nn.Module):

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            raise ExpectedException()
    model = nn.Sequential(Pass(), Pass(), Raise())
    model = Pipe(model, chunks=3)
    with pytest.raises(ExpectedException):
        model(torch.rand(3))

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='2 cuda devices required')
def test_tuple_wait(cuda_sleep, setup_rpc):
    if False:
        i = 10
        return i + 15

    class Sleep(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            if False:
                print('Hello World!')
            return x.detach()

        @staticmethod
        def backward(ctx, grad):
            if False:
                print('Hello World!')
            with torch.cuda.device(grad.device):
                cuda_sleep(0.05)
            return grad

    class Layer1(nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.ones = nn.Parameter(torch.ones(32, 3, 32, 32, requires_grad=True))

        def forward(self, a, b):
            if False:
                while True:
                    i = 10
            a = a * self.ones
            return (a * 1, b * 2, b * 3)

    class Layer2(nn.Module):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.ones = nn.Parameter(torch.ones(32, 3, 32, 32, requires_grad=True))

        def forward(self, a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            a = a * self.ones
            b = Sleep.apply(b)
            return a + b + c
    model = nn.Sequential(Layer1().cuda(0), Layer2().cuda(1))
    model = Pipe(model, chunks=32, checkpoint='never')
    a = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)
    b = torch.rand(1024, 3, 32, 32, device=0, requires_grad=True)
    y = model(a, b)
    y.local_value().norm().backward()
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert torch.isclose(b.grad.norm().cpu(), torch.tensor(5.0))

def test_parallel_randoms(setup_rpc):
    if False:
        for i in range(10):
            print('nop')

    class Dropouts(nn.Module):

        def forward(self, x):
            if False:
                while True:
                    i = 10
            for _ in range(100):
                x = F.dropout(x, p=0.001)
            return x
    model = nn.Sequential(Dropouts(), Dropouts())
    x = torch.rand(10, 10, requires_grad=True)
    model = Pipe(model, chunks=10, checkpoint='always')
    y = model(x)
    y = y.local_value()
    y.norm().backward()
    assert y.to(torch.bool).tolist() == x.grad.to(torch.bool).tolist()
if __name__ == '__main__':
    run_tests()