from collections import OrderedDict
import pytest
import torch
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import BackboneFinetuning, BaseFinetuning, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from torch import nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from tests_pytorch.helpers.runif import RunIf

class TestBackboneFinetuningCallback(BackboneFinetuning):

    def on_train_epoch_start(self, trainer, pl_module):
        if False:
            print('Hello World!')
        super().on_train_epoch_start(trainer, pl_module)
        epoch = trainer.current_epoch
        if self.unfreeze_backbone_at_epoch <= epoch:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            backbone_lr = self.previous_backbone_lr
            if epoch < 6:
                assert backbone_lr <= current_lr
            else:
                assert backbone_lr == current_lr

def test_finetuning_callback(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test finetuning callbacks works as expected.'
    seed_everything(42)

    class FinetuningBoringModel(BoringModel):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            self.layer = torch.nn.Linear(32, 2)
            self.backbone.has_been_used = False

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            self.backbone.has_been_used = True
            x = self.backbone(x)
            return self.layer(x)

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            return ([optimizer], [lr_scheduler])

        def train_dataloader(self):
            if False:
                for i in range(10):
                    print('nop')
            return DataLoader(RandomDataset(32, 64), batch_size=2)
    model = FinetuningBoringModel()
    callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)
    trainer = Trainer(limit_train_batches=4, default_root_dir=tmpdir, callbacks=[callback], max_epochs=8)
    trainer.fit(model)
    assert model.backbone.has_been_used

class TestBackboneFinetuningWarningCallback(BackboneFinetuning):

    def finetune_function(self, pl_module, epoch: int, optimizer):
        if False:
            i = 10
            return i + 15
        'Called when the epoch begins.'
        if epoch == 0:
            self.unfreeze_and_add_param_group(pl_module.backbone, optimizer, 0.1, train_bn=self.train_bn, initial_denom_lr=self.initial_denom_lr)

def test_finetuning_callback_warning(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test finetuning callbacks works as expected.'
    seed_everything(42)

    class FinetuningBoringModel(BoringModel):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.backbone = nn.Linear(32, 2, bias=False)
            self.layer = None
            self.backbone.has_been_used = False

        def forward(self, x):
            if False:
                i = 10
                return i + 15
            self.backbone.has_been_used = True
            return self.backbone(x)

        def train_dataloader(self):
            if False:
                for i in range(10):
                    print('nop')
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def configure_optimizers(self):
            if False:
                return 10
            return torch.optim.SGD(self.parameters(), lr=0.1)
    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = FinetuningBoringModel()
    model.validation_step = None
    callback = TestBackboneFinetuningWarningCallback(unfreeze_backbone_at_epoch=3, verbose=False)
    with pytest.warns(UserWarning, match='Did you init your optimizer in'):
        trainer = Trainer(limit_train_batches=1, default_root_dir=tmpdir, callbacks=[callback, chk], max_epochs=2)
        trainer.fit(model)
    assert model.backbone.has_been_used
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, ckpt_path=chk.last_model_path)

def test_freeze_unfreeze_function(tmpdir):
    if False:
        print('Hello World!')
    'Test freeze properly sets requires_grad on the modules.'
    seed_everything(42)

    class FreezeModel(LightningModule):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))
    model = FreezeModel()
    assert model.backbone[1].track_running_stats
    BaseFinetuning.freeze(model, train_bn=True)
    assert not model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad
    BaseFinetuning.freeze(model, train_bn=False)
    assert not model.backbone[0].weight.requires_grad
    assert not model.backbone[1].weight.requires_grad
    assert not model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad
    BaseFinetuning.make_trainable(model)
    assert model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert model.backbone[3].weight.requires_grad
    BaseFinetuning.freeze(model.backbone[0], train_bn=False)
    assert not model.backbone[0].weight.requires_grad
    BaseFinetuning.freeze([model.backbone[1], [model.backbone[3]]], train_bn=True)
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad

def test_unfreeze_and_add_param_group_function(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test unfreeze_and_add_param_group properly unfreeze parameters and add to the correct param_group.'
    seed_everything(42)

    class FreezeModel(LightningModule):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32))
    model = FreezeModel()
    optimizer = SGD(model.backbone[0].parameters(), lr=0.01)
    with pytest.warns(UserWarning, match='The provided params to be frozen already'):
        BaseFinetuning.unfreeze_and_add_param_group(model.backbone[0], optimizer=optimizer)
    assert optimizer.param_groups[0]['lr'] == 0.01
    model.backbone[1].weight.requires_grad = False
    BaseFinetuning.unfreeze_and_add_param_group(model.backbone[1], optimizer=optimizer)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]['lr'] == 0.001
    assert torch.equal(optimizer.param_groups[1]['params'][0], model.backbone[1].weight)
    assert model.backbone[1].weight.requires_grad
    with pytest.warns(UserWarning, match='The provided params to be frozen already'):
        BaseFinetuning.unfreeze_and_add_param_group(model, optimizer=optimizer, lr=100, train_bn=False)
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[2]['lr'] == 100
    assert len(optimizer.param_groups[2]['params']) == 3
    for (group_idx, group) in enumerate(optimizer.param_groups):
        if group_idx == 0:
            assert torch.equal(optimizer.param_groups[0]['params'][0], model.backbone[0].weight)
        if group_idx == 2:
            assert torch.equal(optimizer.param_groups[2]['params'][0], model.backbone[2].weight)
            assert torch.equal(optimizer.param_groups[2]['params'][1], model.backbone[3].weight)
            assert torch.equal(optimizer.param_groups[2]['params'][2], model.backbone[4].weight)

class OnEpochLayerFinetuning(BaseFinetuning):

    def freeze_before_training(self, pl_module: LightningModule):
        if False:
            i = 10
            return i + 15
        self.freeze(pl_module.layer)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        if False:
            print('Hello World!')
        self.unfreeze_and_add_param_group(pl_module.layer[epoch + 1], optimizer)

def test_base_finetuning_internal_optimizer_metadata(tmpdir):
    if False:
        while True:
            i = 10
    'Test the param_groups updates are properly saved within the internal state of the BaseFinetuning Callbacks.'
    seed_everything(42)

    class FreezeModel(BoringModel):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.layer = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=True), nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=True), nn.Linear(32, 32, bias=False), nn.Linear(32, 2, bias=True))

        def forward(self, x):
            if False:
                print('Hello World!')
            return self.layer(x)

        def configure_optimizers(self):
            if False:
                print('Hello World!')
            return torch.optim.SGD(self.layer[0].parameters(), lr=0.1)
    cb = OnEpochLayerFinetuning()
    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = FreezeModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=5, limit_train_batches=1, callbacks=[cb, chk])
    trainer.fit(model)
    assert len(cb._internal_optimizer_metadata[0]) == 6
    assert cb._internal_optimizer_metadata[0][0]['params'] == ['layer.0.weight']
    assert cb._internal_optimizer_metadata[0][1]['params'] == ['layer.1.weight', 'layer.1.bias']
    assert cb._internal_optimizer_metadata[0][2]['params'] == ['layer.2.weight']
    assert cb._internal_optimizer_metadata[0][3]['params'] == ['layer.3.weight', 'layer.3.bias']
    assert cb._internal_optimizer_metadata[0][4]['params'] == ['layer.4.weight']
    assert cb._internal_optimizer_metadata[0][5]['params'] == ['layer.5.weight', 'layer.5.bias']
    model = FreezeModel()
    cb = OnEpochLayerFinetuning()
    trainer = Trainer(max_epochs=10, callbacks=[cb])
    with pytest.raises(IndexError, match='index 6 is out of range'):
        trainer.fit(model, ckpt_path=chk.last_model_path)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.conv(x)
        x = self.act(x)
        return self.bn(x)

class ConvBlockParam(nn.Module):

    def __init__(self, in_channels, out_channels):
        if False:
            print('Hello World!')
        super().__init__()
        self.module_dict = nn.ModuleDict({'conv': nn.Conv2d(in_channels, out_channels, 3), 'act': nn.ReLU()})
        self.parent_param = nn.Parameter(torch.zeros(1, dtype=torch.float))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if False:
            return 10
        x = self.module_dict['conv'](x)
        x = self.module_dict['act'](x)
        return self.bn(x)

def test_complex_nested_model():
    if False:
        i = 10
        return i + 15
    'Test flattening, freezing, and thawing of models which contain parent (non-leaf) modules with parameters\n    directly themselves rather than exclusively their submodules containing parameters.'
    model = nn.Sequential(OrderedDict([('encoder', nn.Sequential(ConvBlockParam(3, 64), ConvBlock(64, 128))), ('decoder', ConvBlock(128, 10))]))
    assert len(BaseFinetuning.flatten_modules(model)) == 10
    BaseFinetuning.freeze(model.encoder, train_bn=True)
    assert not model.encoder[0].module_dict['conv'].weight.requires_grad
    assert not model.encoder[0].parent_param.requires_grad
    assert model.encoder[0].bn.weight.requires_grad
    BaseFinetuning.make_trainable(model)
    encoder_params = list(BaseFinetuning.filter_params(model.encoder, train_bn=True))
    assert len(encoder_params) == 9

class TestCallbacksRestoreCallback(BaseFinetuning):

    def freeze_before_training(self, pl_module):
        if False:
            return 10
        self.freeze(pl_module.layer[:3])

    def finetune_function(self, pl_module, epoch, optimizer):
        if False:
            print('Hello World!')
        if epoch >= 1:
            self.unfreeze_and_add_param_group(pl_module.layer[epoch - 1], optimizer)

class FinetuningBoringModel(BoringModel):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 2))

    def configure_optimizers(self):
        if False:
            return 10
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        return torch.optim.SGD(parameters, lr=0.1)

def test_callbacks_restore(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test callbacks restore is called after optimizers have been re-created but before optimizer states reload.'
    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = FinetuningBoringModel()
    callback = TestCallbacksRestoreCallback()
    trainer_kwargs = {'default_root_dir': tmpdir, 'limit_train_batches': 1, 'limit_val_batches': 1, 'callbacks': [callback, chk], 'max_epochs': 2}
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)
    assert len(callback._internal_optimizer_metadata) == 1
    assert len(callback._internal_optimizer_metadata[0]) == 2
    expected = {'lr': 0.1, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': ['layer.3.weight', 'layer.3.bias'], 'maximize': False, 'foreach': None}
    if _TORCH_GREATER_EQUAL_1_13:
        expected['differentiable'] = False
    assert callback._internal_optimizer_metadata[0][0] == expected
    expected = {'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': ['layer.0.weight', 'layer.0.bias'], 'maximize': False, 'foreach': None}
    if _TORCH_GREATER_EQUAL_1_13:
        expected['differentiable'] = False
    assert callback._internal_optimizer_metadata[0][1] == expected
    trainer_kwargs['max_epochs'] = 3
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, ckpt_path=chk.last_model_path)

class BackboneBoringModel(BoringModel):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.layer = nn.Linear(32, 2)
        self.backbone = nn.Linear(32, 32)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.layer(self.backbone(x))

def test_callbacks_restore_backbone(tmpdir):
    if False:
        return 10
    'Test callbacks restore is called after optimizers have been re-created but before optimizer states reload.'
    ckpt = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=2, enable_progress_bar=False, callbacks=[ckpt, BackboneFinetuning(unfreeze_backbone_at_epoch=1)])
    trainer.fit(BackboneBoringModel())
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, max_epochs=3, enable_progress_bar=False, callbacks=BackboneFinetuning(unfreeze_backbone_at_epoch=1))
    trainer.fit(BackboneBoringModel(), ckpt_path=ckpt.last_model_path)

@RunIf(deepspeed=True)
def test_unsupported_strategies(tmp_path):
    if False:
        print('Hello World!')
    model = BackboneBoringModel()
    callback = BackboneFinetuning()
    trainer = Trainer(accelerator='cpu', strategy='deepspeed', callbacks=[callback])
    with pytest.raises(NotImplementedError, match='does not support running with the DeepSpeed strategy'):
        callback.setup(trainer, model, stage=None)