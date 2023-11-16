from copy import deepcopy
from unittest.mock import Mock
import torch
from lightning.fabric import Fabric
from lightning.pytorch.demos.boring_classes import BoringModel, ManualOptimBoringModel

def test_fabric_boring_lightning_module_automatic():
    if False:
        for i in range(10):
            print('nop')
    "Test that basic LightningModules written for 'automatic optimization' work with Fabric."
    fabric = Fabric(accelerator='cpu', devices=1)
    module = BoringModel()
    parameters_before = deepcopy(list(module.parameters()))
    (optimizers, _) = module.configure_optimizers()
    dataloader = module.train_dataloader()
    (model, optimizer) = fabric.setup(module, optimizers[0])
    dataloader = fabric.setup_dataloaders(dataloader)
    batch = next(iter(dataloader))
    output = model.training_step(batch, 0)
    fabric.backward(output['loss'])
    optimizer.step()
    assert all((not torch.equal(before, after) for (before, after) in zip(parameters_before, model.parameters())))

def test_fabric_boring_lightning_module_manual():
    if False:
        print('Hello World!')
    "Test that basic LightningModules written for 'manual optimization' work with Fabric."
    fabric = Fabric(accelerator='cpu', devices=1)
    module = ManualOptimBoringModel()
    parameters_before = deepcopy(list(module.parameters()))
    (optimizers, _) = module.configure_optimizers()
    dataloader = module.train_dataloader()
    (model, optimizer) = fabric.setup(module, optimizers[0])
    dataloader = fabric.setup_dataloaders(dataloader)
    batch = next(iter(dataloader))
    model.training_step(batch, 0)
    assert all((not torch.equal(before, after) for (before, after) in zip(parameters_before, model.parameters())))

def test_fabric_call_lightning_module_hooks():
    if False:
        while True:
            i = 10
    'Test that `Fabric.call` can call hooks on the LightningModule.'

    class HookedModel(BoringModel):

        def on_train_start(self):
            if False:
                print('Hello World!')
            pass

        def on_my_custom_hook(self, arg, kwarg=None):
            if False:
                i = 10
                return i + 15
            pass
    fabric = Fabric(accelerator='cpu', devices=1)
    module = Mock(wraps=HookedModel())
    _ = fabric.setup(module)
    _ = fabric.setup(module)
    assert fabric._callbacks == [module]
    fabric.call('on_train_start')
    module.on_train_start.assert_called_once_with()
    fabric.call('on_my_custom_hook', 1, kwarg='test')
    module.on_my_custom_hook.assert_called_once_with(1, kwarg='test')