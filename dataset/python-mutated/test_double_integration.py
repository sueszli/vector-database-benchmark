"""Integration tests for double-precision training."""
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from tests_fabric.helpers.runif import RunIf

class BoringDoubleModule(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.register_buffer('complex_buffer', torch.complex(torch.rand(10), torch.rand(10)), False)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        assert x.dtype == torch.float64
        assert torch.tensor([0.0]).dtype == torch.float64
        return self.layer(x)

@RunIf(mps=False)
def test_double_precision():
    if False:
        print('Hello World!')
    fabric = Fabric(devices=1, precision='64-true')
    with fabric.init_module():
        model = BoringDoubleModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    (model, optimizer) = fabric.setup(model, optimizer)
    batch = torch.rand(2, 32, device=fabric.device)
    assert model.layer.weight.dtype == model.layer.bias.dtype == torch.float64
    assert model.complex_buffer.dtype == torch.complex128
    assert batch.dtype == torch.float32
    output = model(batch)
    assert output.dtype == torch.float32
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    fabric.backward(loss)
    assert model.layer.weight.grad.dtype == torch.float64
    optimizer.step()
    optimizer.zero_grad()