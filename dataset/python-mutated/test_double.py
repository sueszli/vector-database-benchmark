import torch
from lightning.fabric.plugins.precision.double import DoublePrecision

def test_double_precision_forward_context():
    if False:
        while True:
            i = 10
    precision = DoublePrecision()
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == torch.float32

def test_convert_module():
    if False:
        print('Hello World!')
    precision = DoublePrecision()
    module = torch.nn.Linear(2, 2)
    assert module.weight.dtype == module.bias.dtype == torch.float32
    module = precision.convert_module(module)
    assert module.weight.dtype == module.bias.dtype == torch.float64