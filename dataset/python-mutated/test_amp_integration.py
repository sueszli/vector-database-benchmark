"""Integration tests for Automatic Mixed Precision (AMP) training."""
import pytest
import torch
import torch.nn as nn
from lightning.fabric import Fabric, seed_everything
from tests_fabric.helpers.runif import RunIf

class MixedPrecisionModule(nn.Module):

    def __init__(self, expected_dtype):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.expected_dtype = expected_dtype
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        if False:
            print('Hello World!')
        assert x.dtype == self.expected_dtype
        if x.device.type == 'cpu':
            assert torch.is_autocast_cpu_enabled()
        else:
            assert torch.is_autocast_enabled()
        output = self.layer(x)
        assert output.dtype == self.expected_dtype
        return output

@pytest.mark.parametrize(('accelerator', 'precision', 'expected_dtype'), [('cpu', '16-mixed', torch.bfloat16), ('cpu', 'bf16-mixed', torch.bfloat16), pytest.param('cuda', '16-mixed', torch.float16, marks=RunIf(min_cuda_gpus=2)), pytest.param('cuda', 'bf16-mixed', torch.bfloat16, marks=RunIf(min_cuda_gpus=2, bf16_cuda=True))])
def test_amp(accelerator, precision, expected_dtype):
    if False:
        i = 10
        return i + 15
    fabric = Fabric(accelerator=accelerator, precision=precision, devices=2, strategy='ddp_spawn')
    fabric.launch(_test_amp, expected_dtype)

def _test_amp(fabric, expected_dtype):
    if False:
        while True:
            i = 10
    model = MixedPrecisionModule(expected_dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    (model, optimizer) = fabric.setup(model, optimizer)
    batch = torch.rand(2, 32, device=fabric.device)
    assert model.layer.weight.dtype == torch.float32
    assert batch.dtype == torch.float32
    output = model(batch)
    assert output.dtype == torch.float32
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    fabric.backward(loss)
    assert model.layer.weight.grad.dtype == torch.float32
    optimizer.step()
    optimizer.zero_grad()

@RunIf(min_torch='1.13', min_cuda_gpus=1)
def test_amp_fused_optimizer_parity():
    if False:
        for i in range(10):
            print('nop')

    def run(fused=False):
        if False:
            return 10
        seed_everything(1234)
        fabric = Fabric(accelerator='cuda', precision=16, devices=1)
        model = nn.Linear(10, 10).to(fabric.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, fused=fused)
        (model, optimizer) = fabric.setup(model, optimizer)
        assert isinstance(fabric._precision.scaler, torch.cuda.amp.GradScaler)
        data = torch.randn(10, 10, device='cuda')
        target = torch.randn(10, 10, device='cuda')
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(data)
            loss = (output - target).abs().sum()
            fabric.backward(loss)
            optimizer.step()
            losses.append(loss.detach())
        return (torch.stack(losses), model.parameters())
    (losses, params) = run(fused=False)
    (losses_fused, params_fused) = run(fused=True)
    torch.testing.assert_close(losses, losses_fused)
    for (p, q) in zip(params, params_fused):
        torch.testing.assert_close(p, q)