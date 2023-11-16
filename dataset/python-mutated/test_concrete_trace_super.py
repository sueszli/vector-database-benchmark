import unittest
import torch
import torch.nn.functional as F
from nni.common.concrete_trace_utils import concrete_trace

class Foo(torch.nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return F.relu(x)

class Bar1(Foo):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return super(Bar1, self).forward(x)

class Bar2(Foo):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        return super(self.__class__, self).forward(x)

class Bar3(Foo):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return super().forward(x)

class TraceSuperCase(unittest.TestCase):

    def the_test(self, model):
        if False:
            for i in range(10):
                print('nop')
        dummy_input = torch.rand(8, 1, 28, 28)
        traced_model = concrete_trace(model, {'x': dummy_input})
        print(traced_model)

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        return self.the_test(Bar1())

    def test_2(self):
        if False:
            while True:
                i = 10
        return self.the_test(Bar2())

    def test_3(self):
        if False:
            i = 10
            return i + 15
        return self.the_test(Bar3())
if __name__ == '__main__':
    unittest.main()