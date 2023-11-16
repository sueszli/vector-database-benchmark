import pytest
import torch
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision

def test_invalid_precision_with_deepspeed_precision():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='is not supported. `precision` must be one of'):
        DeepSpeedPrecision(precision='64-true')

@pytest.mark.parametrize(('precision', 'expected_dtype'), [('32-true', torch.float32), ('bf16-mixed', torch.bfloat16), ('16-mixed', torch.float16), ('bf16-true', torch.bfloat16), ('16-true', torch.float16)])
def test_selected_dtype(precision, expected_dtype):
    if False:
        while True:
            i = 10
    plugin = DeepSpeedPrecision(precision=precision)
    assert plugin.precision == precision
    assert plugin._desired_dtype == expected_dtype

@pytest.mark.parametrize(('precision', 'expected_dtype'), [('32-true', torch.float32), ('bf16-mixed', torch.float32), ('16-mixed', torch.float32), ('bf16-true', torch.bfloat16), ('16-true', torch.float16)])
def test_module_init_context(precision, expected_dtype):
    if False:
        i = 10
        return i + 15
    plugin = DeepSpeedPrecision(precision=precision)
    with plugin.module_init_context():
        model = torch.nn.Linear(2, 2)
        assert torch.get_default_dtype() == expected_dtype
    assert model.weight.dtype == expected_dtype

@pytest.mark.parametrize(('precision', 'expected_dtype'), [('32-true', torch.float32), ('bf16-mixed', torch.float32), ('16-mixed', torch.float32), ('bf16-true', torch.bfloat16), ('16-true', torch.float16)])
def test_convert_module(precision, expected_dtype):
    if False:
        return 10
    precision = DeepSpeedPrecision(precision=precision)
    module = torch.nn.Linear(2, 2)
    assert module.weight.dtype == module.bias.dtype == torch.float32
    module = precision.convert_module(module)
    assert module.weight.dtype == module.bias.dtype == expected_dtype