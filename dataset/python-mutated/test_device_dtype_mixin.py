import pytest
import torch
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from torch import nn as nn
from tests_fabric.helpers.runif import RunIf

class SubSubModule(_DeviceDtypeModuleMixin):
    pass

class SubModule(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.module = SubSubModule()

class TopModule(_DeviceDtypeModuleMixin):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.module = SubModule()

@pytest.mark.parametrize(('dst_device_str', 'dst_type'), [('cpu', torch.half), ('cpu', torch.float), ('cpu', torch.double), pytest.param('cuda:0', torch.half, marks=RunIf(min_cuda_gpus=1)), pytest.param('cuda:0', torch.float, marks=RunIf(min_cuda_gpus=1)), pytest.param('cuda:0', torch.double, marks=RunIf(min_cuda_gpus=1)), pytest.param('mps:0', torch.float, marks=RunIf(mps=True))])
@RunIf(min_cuda_gpus=1)
def test_submodules_device_and_dtype(dst_device_str, dst_type):
    if False:
        i = 10
        return i + 15
    'Test that the device and dtype property updates propagate through mixed nesting of regular nn.Modules and the\n    special modules of type DeviceDtypeModuleMixin (e.g. Metric or LightningModule).'
    dst_device = torch.device(dst_device_str)
    model = TopModule()
    assert model.device == torch.device('cpu')
    model = model.to(device=dst_device, dtype=dst_type)
    assert not hasattr(model.module, '_device')
    assert not hasattr(model.module, '_dtype')
    assert model.device == model.module.module.device == dst_device
    assert model.dtype == model.module.module.dtype == dst_type

@pytest.mark.parametrize('device', [None, 0, torch.device('cuda', 0)])
@RunIf(min_cuda_gpus=1)
def test_cuda_device(device):
    if False:
        while True:
            i = 10
    model = TopModule()
    model.cuda(device)
    device = model.device
    assert device.type == 'cuda'
    assert device.index is not None
    assert device.index == torch.cuda.current_device()

@RunIf(min_cuda_gpus=1)
def test_cpu_device():
    if False:
        for i in range(10):
            print('nop')
    model = SubSubModule().cuda()
    assert model.device.type == 'cuda'
    assert model.device.index == 0
    model.cpu()
    assert model.device.type == 'cpu'
    assert model.device.index is None

@RunIf(min_cuda_gpus=2)
def test_cuda_current_device():
    if False:
        print('Hello World!')
    'Test that calling .cuda() moves the model to the correct device and respects current cuda device setting.'

    class CudaModule(_DeviceDtypeModuleMixin):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.layer = nn.Linear(1, 1)
    model = CudaModule()
    torch.cuda.set_device(0)
    model.cuda(1)
    assert model.device == torch.device('cuda', 1)
    assert model.layer.weight.device == torch.device('cuda', 1)
    torch.cuda.set_device(1)
    model.cuda()
    assert model.device == torch.device('cuda', 1)
    assert model.layer.weight.device == torch.device('cuda', 1)

class ExampleModule(_DeviceDtypeModuleMixin):

    def __init__(self, weight):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.register_buffer('weight', weight)

def test_to_combinations():
    if False:
        print('Hello World!')
    module = ExampleModule(torch.rand(3, 4))
    assert module.weight.shape == (3, 4)
    assert module.weight.dtype is torch.float32
    module.to(torch.double)
    assert module.weight.dtype is torch.float64
    module.to('cpu', dtype=torch.half, non_blocking=True)
    assert module.weight.dtype is torch.float16
    assert module.device == torch.device('cpu')
    assert module.dtype is torch.float16

def test_dtype_conversions():
    if False:
        for i in range(10):
            print('nop')
    module = ExampleModule(torch.tensor(1))
    assert module.weight.dtype is torch.int64
    assert module.dtype is torch.float32
    module.double()
    assert module.weight.dtype is torch.int64
    assert module.dtype is torch.float64
    module.type(torch.float)
    assert module.weight.dtype is torch.float32
    assert module.dtype is torch.float32
    module.float()
    assert module.weight.dtype is torch.float32
    assert module.dtype is torch.float32
    module.half()
    assert module.weight.dtype is torch.float16
    assert module.dtype is torch.float16