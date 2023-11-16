import sys
from unittest.mock import Mock
import lightning.fabric
import pytest
import torch.nn
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision.double import LightningDoublePrecisionModule
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from tests_pytorch.helpers.runif import RunIf

def test_configure_sharded_model():
    if False:
        for i in range(10):
            print('nop')

    class MyModel(BoringModel):

        def configure_sharded_model(self) -> None:
            if False:
                print('Hello World!')
            ...
    model = MyModel()
    trainer = Trainer(devices=1, accelerator='cpu', fast_dev_run=1)
    with pytest.deprecated_call(match='overridden `MyModel.configure_sharded_model'):
        trainer.fit(model)

    class MyModelBoth(MyModel):

        def configure_model(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
    model = MyModelBoth()
    with pytest.raises(RuntimeError, match='Both `MyModelBoth.configure_model`, and `MyModelBoth.configure_sharded_model`'):
        trainer.fit(model)

def test_ddp_is_distributed():
    if False:
        while True:
            i = 10
    strategy = DDPStrategy()
    with pytest.deprecated_call(match='is deprecated'):
        _ = strategy.is_distributed

@RunIf(min_torch='1.13')
def test_fsdp_activation_checkpointing(monkeypatch):
    if False:
        return 10
    with pytest.raises(ValueError, match='cannot set both `activation_checkpointing'):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear, activation_checkpointing_policy=lambda *_: True)
    monkeypatch.setattr(lightning.fabric.strategies.fsdp, '_TORCH_GREATER_EQUAL_2_1', True)
    with pytest.deprecated_call(match='use `FSDPStrategy\\(activation_checkpointing_policy'):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear)

def test_double_precision_wrapper():
    if False:
        i = 10
        return i + 15
    with pytest.deprecated_call(match='The `LightningDoublePrecisionModule` is deprecated and no longer needed'):
        LightningDoublePrecisionModule(BoringModel())

def test_fsdp_mixed_precision_plugin():
    if False:
        return 10
    from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin
    with pytest.deprecated_call(match='The `FSDPMixedPrecisionPlugin` is deprecated'):
        FSDPMixedPrecisionPlugin(precision='16-mixed', device='cuda')

def test_fsdp_precision_plugin():
    if False:
        while True:
            i = 10
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin
    with pytest.deprecated_call(match='The `FSDPPrecisionPlugin` is deprecated'):
        FSDPPrecisionPlugin(precision='16-mixed')

def test_bitsandbytes_precision_plugin(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(lightning.fabric.plugins.precision.bitsandbytes, '_BITSANDBYTES_AVAILABLE', True)
    bitsandbytes_mock = Mock()
    monkeypatch.setitem(sys.modules, 'bitsandbytes', bitsandbytes_mock)
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin
    with pytest.deprecated_call(match='The `BitsandbytesPrecisionPlugin` is deprecated'):
        BitsandbytesPrecisionPlugin('nf4')

def test_deepspeed_precision_plugin():
    if False:
        print('Hello World!')
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
    with pytest.deprecated_call(match='The `DeepSpeedPrecisionPlugin` is deprecated'):
        DeepSpeedPrecisionPlugin(precision='32-true')

def test_double_precision_plugin():
    if False:
        while True:
            i = 10
    from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin
    with pytest.deprecated_call(match='The `DoublePrecisionPlugin` is deprecated'):
        DoublePrecisionPlugin()

def test_half_precision_plugin():
    if False:
        return 10
    from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin
    with pytest.deprecated_call(match='The `HalfPrecisionPlugin` is deprecated'):
        HalfPrecisionPlugin()

def test_mixed_precision_plugin():
    if False:
        return 10
    from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin
    with pytest.deprecated_call(match='The `MixedPrecisionPlugin` is deprecated'):
        MixedPrecisionPlugin(precision='16-mixed', device='cuda')

def test_precision_plugin():
    if False:
        return 10
    from lightning.pytorch.plugins.precision.precision import PrecisionPlugin
    with pytest.deprecated_call(match='The `PrecisionPlugin` is deprecated'):
        PrecisionPlugin()

def test_transformer_engine_precision_plugin(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(lightning.fabric.plugins.precision.transformer_engine, '_TRANSFORMER_ENGINE_AVAILABLE', True)
    transformer_engine_mock = Mock()
    monkeypatch.setitem(sys.modules, 'transformer_engine', transformer_engine_mock)
    monkeypatch.setitem(sys.modules, 'transformer_engine.pytorch', Mock())
    recipe_mock = Mock()
    monkeypatch.setitem(sys.modules, 'transformer_engine.common.recipe', recipe_mock)
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin
    with pytest.deprecated_call(match='The `TransformerEnginePrecisionPlugin` is deprecated'):
        TransformerEnginePrecisionPlugin()

def test_xla_precision_plugin(xla_available):
    if False:
        return 10
    from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin
    with pytest.deprecated_call(match='The `XLAPrecisionPlugin` is deprecated'):
        XLAPrecisionPlugin()