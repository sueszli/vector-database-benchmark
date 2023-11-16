import os
import re
from unittest import mock
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins import MixedPrecision
from torch import Tensor
from tests_pytorch.helpers.runif import RunIf

class MyAMP(MixedPrecision):
    pass

@RunIf(mps=False)
@mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'SLURM_NTASKS': '2', 'SLURM_NTASKS_PER_NODE': '1', 'SLURM_JOB_NAME': 'SOME_NAME', 'SLURM_NODEID': '0', 'LOCAL_RANK': '0', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0'})
@pytest.mark.parametrize(('strategy', 'devices'), [('ddp', 2), ('ddp_spawn', 2)])
@pytest.mark.parametrize(('custom_plugin', 'plugin_cls'), [(False, MixedPrecision), (True, MyAMP)])
def test_amp_ddp(cuda_count_2, strategy, devices, custom_plugin, plugin_cls):
    if False:
        for i in range(10):
            print('nop')
    plugin = None
    precision = None
    if custom_plugin:
        plugin = plugin_cls('16-mixed', 'cpu')
    else:
        precision = '16-mixed'
    trainer = Trainer(fast_dev_run=True, precision=precision, accelerator='gpu', devices=devices, strategy=strategy, plugins=plugin)
    assert isinstance(trainer.precision_plugin, plugin_cls)

class TestClippingOptimizer(torch.optim.SGD):

    def step(self, *args, pl_module=None):
        if False:
            i = 10
            return i + 15
        pl_module.check_grads_clipped()
        return super().step(*args)

class TestPrecisionModel(BoringModel):

    def on_after_backward(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        scale = self.trainer.precision_plugin.scaler.get_scale()
        assert scale != 1.0
        grads = [p.grad for p in self.parameters()]
        inv_scale = 1 / scale
        self.original_grads = [p * inv_scale for p in grads]

    def check_grads_unscaled(self, optimizer=None):
        if False:
            print('Hello World!')
        if optimizer is not None:
            scaler = self.trainer.precision_plugin.scaler
            state = scaler._per_optimizer_states[id(optimizer)]
            assert state['stage'].name == 'UNSCALED'
        grads = [p.grad for p in self.parameters()]
        assert len(grads) == len(self.original_grads)
        for (actual, expected) in zip(grads, self.original_grads):
            torch.testing.assert_close(actual, expected, equal_nan=True)

    def check_grads_clipped(self):
        if False:
            for i in range(10):
                print('nop')
        parameters = list(self.parameters())
        assert len(parameters) == len(self.clipped_parameters)
        for (actual, expected) in zip(parameters, self.clipped_parameters):
            torch.testing.assert_close(actual.grad, expected.grad, equal_nan=True)

    def on_before_optimizer_step(self, optimizer, *_):
        if False:
            i = 10
            return i + 15
        self.check_grads_unscaled(optimizer)
        self.clipped_parameters = []
        for p in self.parameters():
            copy = p.detach().clone()
            copy.grad = p.grad.clone()
            self.clipped_parameters.append(copy)
        clip_val = self.trainer.gradient_clip_val
        torch.nn.utils.clip_grad_value_(self.clipped_parameters, clip_val)

    def configure_gradient_clipping(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().configure_gradient_clipping(*args, **kwargs)
        self.check_grads_clipped()

    def optimizer_step(self, epoch, batch_idx, optimizer, closure, **_):
        if False:
            i = 10
            return i + 15
        optimizer.step(closure, pl_module=self)

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        return TestClippingOptimizer(self.layer.parameters(), lr=0.1)

@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize('accum', [1, 2])
def test_amp_gradient_unscale(tmpdir, accum: int):
    if False:
        return 10
    model = TestPrecisionModel()
    trainer = Trainer(max_epochs=2, default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=0, strategy='ddp_spawn', accelerator='gpu', devices=2, precision='16-mixed', gradient_clip_val=0.001, gradient_clip_algorithm='value', log_every_n_steps=1, accumulate_grad_batches=accum, enable_progress_bar=False)
    trainer.fit(model)

@RunIf(min_cuda_gpus=1)
def test_amp_skip_optimizer(tmpdir):
    if False:
        return 10
    'Test that optimizers can be skipped when using amp.'

    class CustomBoringModel(BoringModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.automatic_optimization = False
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 2)

        def forward(self, x: Tensor):
            if False:
                for i in range(10):
                    print('nop')
            x = self.layer1(x)
            x = self.layer2(x)
            return x

        def training_step(self, batch, batch_idx):
            if False:
                return 10
            (opt1, opt2) = self.optimizers()
            output = self(batch)
            loss = self.loss(output)
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            return [torch.optim.SGD(self.layer1.parameters(), lr=0.1), torch.optim.SGD(self.layer2.parameters(), lr=0.1)]
    trainer = Trainer(default_root_dir=tmpdir, accelerator='gpu', devices=1, fast_dev_run=1, precision='16-mixed')
    model = CustomBoringModel()
    trainer.fit(model)

def test_cpu_amp_precision_context_manager():
    if False:
        while True:
            i = 10
    'Test to ensure that the context manager correctly is set to CPU + bfloat16.'
    plugin = MixedPrecision('bf16-mixed', 'cpu')
    assert plugin.device == 'cpu'
    assert plugin.scaler is None
    context_manager = plugin.autocast_context_manager()
    assert isinstance(context_manager, torch.autocast)
    assert context_manager.fast_dtype == torch.bfloat16

def test_amp_precision_plugin_parameter_validation():
    if False:
        while True:
            i = 10
    MixedPrecision('16-mixed', 'cpu')
    MixedPrecision('bf16-mixed', 'cpu')
    with pytest.raises(ValueError, match=re.escape("Passed `MixedPrecision(precision='16')`. Precision must be '16-mixed' or 'bf16-mixed'")):
        MixedPrecision('16', 'cpu')
    with pytest.raises(ValueError, match=re.escape("Passed `MixedPrecision(precision=16)`. Precision must be '16-mixed' or 'bf16-mixed'")):
        MixedPrecision(16, 'cpu')
    with pytest.raises(ValueError, match=re.escape("Passed `MixedPrecision(precision='bf16')`. Precision must be '16-mixed' or 'bf16-mixed'")):
        MixedPrecision('bf16', 'cpu')