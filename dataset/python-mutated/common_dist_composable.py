from typing import Tuple
import torch
import torch.nn as nn

class UnitModule(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            while True:
                i = 10
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(nn.ReLU(), nn.Linear(100, 100, device=device), nn.ReLU())
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.l2(self.seq(self.l1(x)))

class CompositeModel(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            print('Hello World!')
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        if False:
            return 10
        return self.l2(self.u2(self.u1(self.l1(x))))

class UnitParamModule(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            while True:
                i = 10
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(nn.ReLU(), nn.Linear(100, 100, device=device), nn.ReLU())
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return torch.mm(self.seq(self.l(x)), self.p)

class CompositeParamModel(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            return 10
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        if False:
            print('Hello World!')
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)

class FakeSequential(nn.Module):

    def __init__(self, *modules: Tuple[nn.Module, ...]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._module_sequence = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        for module in self._module_sequence:
            x = module(x)
        return x

class NestedSequentialModel(nn.Module):

    def __init__(self, device: torch.device) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.seq1 = nn.Sequential(nn.Linear(1, 1, device=device), FakeSequential(nn.Linear(1, 1, device=device), nn.ReLU(), FakeSequential(nn.Linear(1, 1, device=device)), nn.ReLU()), nn.Linear(1, 2, device=device))
        self.lin = nn.Linear(2, 2, device=device)
        self.seq2 = nn.Sequential(nn.ReLU(), nn.Linear(2, 3, device=device), FakeSequential(nn.Linear(3, 2, bias=False, device=device), nn.Linear(2, 4, bias=False, device=device)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return self.seq2(self.lin(self.seq1(x)))