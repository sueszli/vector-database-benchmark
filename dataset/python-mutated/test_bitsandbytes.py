import sys
from unittest.mock import Mock
import lightning.fabric
import pytest
import torch
import torch.distributed
from lightning.fabric import Fabric
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE, BitsandbytesPrecision
from tests_fabric.helpers.runif import RunIf

@pytest.mark.skipif(_BITSANDBYTES_AVAILABLE, reason='bitsandbytes needs to be unavailable')
def test_bitsandbytes_plugin(monkeypatch):
    if False:
        print('Hello World!')
    module = lightning.fabric.plugins.precision.bitsandbytes
    monkeypatch.setattr(module, '_BITSANDBYTES_AVAILABLE', lambda : True)
    bitsandbytes_mock = Mock()
    monkeypatch.setitem(sys.modules, 'bitsandbytes', bitsandbytes_mock)

    class ModuleMock(torch.nn.Linear):

        def __init__(self, in_features, out_features, bias=True, *_, **__):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(in_features, out_features, bias)
    bitsandbytes_mock.nn.Linear8bitLt = ModuleMock
    bitsandbytes_mock.nn.Linear4bit = ModuleMock
    precision = BitsandbytesPrecision('nf4', dtype=torch.float16)
    connector = _Connector(plugins=precision)
    assert connector.precision is precision
    assert precision.dtype == torch.float16
    assert torch.get_default_dtype() is torch.float32
    with pytest.raises(RuntimeError, match='foo'), precision.module_init_context():
        assert torch.get_default_dtype() is not torch.float32
        raise RuntimeError('foo')
    assert torch.get_default_dtype() is torch.float32

    class SubModule(torch.nn.Module):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.l = torch.nn.Linear(1, 3)

    class MyModule(torch.nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.l1 = torch.nn.Linear(16, 48)
            self.l2 = SubModule()
    _NF4Linear = vars(module)['_NF4Linear']
    _NF4Linear._quantize_weight = Mock()
    with precision.module_init_context():
        assert torch.get_default_dtype() == torch.float16
        model = MyModule()
    assert isinstance(model.l1, _NF4Linear)
    assert isinstance(model.l2.l, _NF4Linear)
    model = precision.convert_module(model)
    assert model.l1.compute_dtype is precision.dtype
    assert model.l2.l.compute_dtype is precision.dtype
    model = MyModule()
    precision.convert_module(model)
    assert isinstance(model.l1, _NF4Linear)
    assert isinstance(model.l2.l, _NF4Linear)
    precision.ignore_modules = {'l2'}
    model = MyModule()
    precision.convert_module(model)
    assert isinstance(model.l1, _NF4Linear)
    assert isinstance(model.l2.l, torch.nn.Linear)
    model = torch.nn.Conv1d(1, 1, 1)
    with pytest.raises(TypeError, match='your model has no Linear'):
        precision.convert_module(model)

@RunIf(min_cuda_gpus=1)
@pytest.mark.skipif(not _BITSANDBYTES_AVAILABLE, reason='bitsandbytes unavailable')
@pytest.mark.parametrize(('args', 'expected'), [(('int8', torch.float16), torch.int8), (('nf4', torch.bfloat16), torch.uint8)])
def test_bitsandbytes_layers(args, expected):
    if False:
        while True:
            i = 10

    class MyModel(torch.nn.Module):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.l = torch.nn.Linear(2, 2)
            self.ln = torch.nn.LayerNorm(2)
    state_dict = MyModel().state_dict()
    fabric = Fabric(devices=1, plugins=BitsandbytesPrecision(*args))
    with fabric.init_module():
        model = MyModel()
    assert model.l.weight.device.type == 'cuda'
    assert model.l.weight.dtype == expected
    model = fabric.setup(model)
    assert model.l.weight.device.type == 'cuda'
    assert model.l.weight.dtype == expected
    weight_before = model.l.weight.data.clone()
    keys = model.load_state_dict(state_dict, strict=True)
    assert not keys.missing_keys
    assert not torch.equal(weight_before, model.l.weight.data)
    assert model.l.weight.device.type == 'cuda'
    assert model.l.weight.dtype == expected
    fabric = Fabric(devices=1, plugins=BitsandbytesPrecision(*args, ignore_modules={'foo'}))
    with pytest.raises(RuntimeError, match='not supported'), fabric.init_module():
        pass
    model = MyModel()
    assert model.l.weight.device.type == 'cpu'
    assert model.l.weight.dtype == torch.float32
    model = fabric.setup(model)
    assert model.l.weight.device.type == 'cuda'
    assert model.l.weight.dtype == expected