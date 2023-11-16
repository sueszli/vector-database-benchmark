import torch
from typing import Union

class TestVersionedDivTensorExampleV7(torch.nn.Module):

    def forward(self, a, b):
        if False:
            while True:
                i = 10
        result_0 = a / b
        result_1 = torch.div(a, b)
        result_2 = a.div(b)
        return (result_0, result_1, result_2)

class TestVersionedLinspaceV7(torch.nn.Module):

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        if False:
            i = 10
            return i + 15
        c = torch.linspace(a, b, steps=5)
        d = torch.linspace(a, b)
        return (c, d)

class TestVersionedLinspaceOutV7(torch.nn.Module):

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        return torch.linspace(a, b, out=out)

class TestVersionedLogspaceV8(torch.nn.Module):

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        if False:
            return 10
        c = torch.logspace(a, b, steps=5)
        d = torch.logspace(a, b)
        return (c, d)

class TestVersionedLogspaceOutV8(torch.nn.Module):

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
        if False:
            print('Hello World!')
        return torch.logspace(a, b, out=out)

class TestVersionedGeluV9(torch.nn.Module):

    def forward(self, x):
        if False:
            return 10
        return torch._C._nn.gelu(x)

class TestVersionedGeluOutV9(torch.nn.Module):

    def forward(self, x):
        if False:
            return 10
        out = torch.zeros_like(x)
        return torch._C._nn.gelu(x, out=out)

class TestVersionedRandomV10(torch.nn.Module):

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = torch.zeros_like(x)
        return out.random_(0, 10)

class TestVersionedRandomFuncV10(torch.nn.Module):

    def forward(self, x):
        if False:
            print('Hello World!')
        out = torch.zeros_like(x)
        return out.random(0, 10)

class TestVersionedRandomOutV10(torch.nn.Module):

    def forward(self, x):
        if False:
            print('Hello World!')
        x = torch.zeros_like(x)
        out = torch.zeros_like(x)
        x.random(0, 10, out=out)
        return out