import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters
from torch import nn

class ParameterSharingModule(BoringModel):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

@pytest.mark.parametrize(('model', 'expected_shared_params'), [(BoringModel, []), (ParameterSharingModule, [['layer_1.weight', 'layer_3.weight']])])
def test_find_shared_parameters(model, expected_shared_params):
    if False:
        print('Hello World!')
    assert expected_shared_params == find_shared_parameters(model())

def test_set_shared_parameters():
    if False:
        print('Hello World!')
    model = ParameterSharingModule()
    set_shared_parameters(model, [['layer_1.weight', 'layer_3.weight']])
    assert torch.all(torch.eq(model.layer_1.weight, model.layer_3.weight))

    class SubModule(nn.Module):

        def __init__(self, layer):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.layer = layer

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return self.layer(x)

    class NestedModule(BoringModel):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.layer = nn.Linear(32, 10, bias=False)
            self.net_a = SubModule(self.layer)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.net_b = SubModule(self.layer)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.net_a(x)
            x = self.layer_2(x)
            x = self.net_b(x)
            return x
    model = NestedModule()
    set_shared_parameters(model, [['layer.weight', 'net_a.layer.weight', 'net_b.layer.weight']])
    assert torch.all(torch.eq(model.net_a.layer.weight, model.net_b.layer.weight))