import pytest
import torch
from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.speedup.dependency import build_channel_dependency

class PatternA(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        if False:
            print('Hello World!')
        return torch.cat((self.conv1(x), self.conv2(x)), dim=0) + torch.cat((self.conv3(x), self.conv4(x)), dim=0)

class PatternB(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        if False:
            print('Hello World!')
        return torch.cat((self.conv1(x), self.conv2(x)), dim=1)

class PatternC(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        if False:
            return 10
        return torch.sum(self.conv1(x), dim=1) + torch.sum(self.conv2(x), dim=1)

def check_sets_all_match(a, b):
    if False:
        for i in range(10):
            print('nop')
    assert len(a) == len(b)
    for s in a:
        s = {node.name for node in s}
        assert s in b
        b.pop(b.index(s))
    assert len(b) == 0

@pytest.mark.parametrize('mod, deps', [(PatternA, [{'conv1', 'conv2'}, {'conv3', 'conv4'}]), (PatternB, []), (PatternC, [])])
def test_channel_dependency(mod, deps):
    if False:
        i = 10
        return i + 15
    model = mod()
    dummy_input = (torch.randn(1, 3, 224, 224),)
    traced = concrete_trace(model, dummy_input)
    dependency = build_channel_dependency(traced)
    check_sets_all_match(dependency, deps)