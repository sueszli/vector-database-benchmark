import math
from unittest.mock import Mock, patch
import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _LIGHTNING_COLOSSALAI_AVAILABLE
if _LIGHTNING_COLOSSALAI_AVAILABLE:
    from lightning_colossalai import ColossalAIStrategy
else:
    ColossalAIStrategy = None

@pytest.mark.parametrize('accumulate_grad_batches', [1, 2, 3])
def test_trainer_accumulate_grad_batches_zero_grad(tmpdir, accumulate_grad_batches):
    if False:
        print('Hello World!')
    with patch('torch.optim.SGD.zero_grad') as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=20, limit_val_batches=1, max_epochs=1, enable_model_summary=False, accumulate_grad_batches=accumulate_grad_batches)
        assert trainer.accumulate_grad_batches == accumulate_grad_batches
        trainer.fit(model)
        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)

@pytest.mark.parametrize(('accumulate_grad_batches', 'expected_call_count'), [({1: 2, 3: 4}, 10 + 5 + 5 + 3), ({0: 2, 2: 1}, 5 + 5 + 10 + 10)])
def test_trainer_accumulate_grad_batches_with_callback(tmpdir, accumulate_grad_batches, expected_call_count):
    if False:
        return 10
    with patch('torch.optim.SGD.zero_grad') as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=10, limit_val_batches=1, max_epochs=4, enable_model_summary=False, callbacks=GradientAccumulationScheduler(accumulate_grad_batches))
        assert trainer.accumulate_grad_batches == 1
        trainer.fit(model)
        assert sum((isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks)) == 1
        assert sgd_zero_grad.call_count == expected_call_count

@pytest.mark.parametrize('scheduling', [{1: 2, -3: 4}, {0: 2, '2': 1}])
def test_invalid_keys_for_grad_accum_scheduler(scheduling):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(MisconfigurationException, match='Epoch should be an int'):
        _ = GradientAccumulationScheduler(scheduling=scheduling)

@pytest.mark.parametrize('scheduling', [{1: 0, 3: 4}, {0: 2, 2: '2'}])
def test_invalid_values_for_grad_accum_scheduler(scheduling):
    if False:
        return 10
    with pytest.raises(MisconfigurationException, match='Accumulation factor should be an int'):
        _ = GradientAccumulationScheduler(scheduling=scheduling)

@pytest.mark.parametrize('strategy_class', [pytest.param(ColossalAIStrategy, marks=pytest.mark.skipif(not _LIGHTNING_COLOSSALAI_AVAILABLE, reason='Requires ColossalAI strategy')), DeepSpeedStrategy])
def test_unsupported_strategies(strategy_class):
    if False:
        for i in range(10):
            print('nop')
    'Test that an error is raised for strategies that require the gradient accumulation factor to be fixed.'
    scheduler = GradientAccumulationScheduler({1: 2})
    model = BoringModel()
    trainer = Trainer()
    trainer._accelerator_connector.strategy = Mock(spec=strategy_class)
    with pytest.raises(RuntimeError, match='does not support `accumulate_grad_batches` changing between epochs'):
        scheduler.on_train_start(trainer, model)

def test_unsupported_manual_optimization():
    if False:
        while True:
            i = 10
    'Test that an error is raised when attempting to use the callback with manual optimization.'
    scheduler = GradientAccumulationScheduler({1: 2})
    model = BoringModel()
    model.automatic_optimization = False
    trainer = Trainer()
    with pytest.raises(RuntimeError, match='Automatic gradient accumulation and the `GradientAccumulationScheduler`'):
        scheduler.on_train_start(trainer, model)

def test_warn_if_model_has_overridden_optimization_hooks():
    if False:
        print('Hello World!')
    'Test that the callback warns if optimization hooks were overridden in the LightningModule.'

    class OverriddenOptimizerStepModel(BoringModel):

        def optimizer_step(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super().optimizer_step(*args, **kwargs)

    class OverriddenZeroGradModel(BoringModel):

        def optimizer_zero_grad(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().optimizer_zero_grad(*args, **kwargs)
    scheduler = GradientAccumulationScheduler({1: 2})
    trainer = Trainer()
    model = OverriddenOptimizerStepModel()
    with pytest.warns(UserWarning, match='the hooks will not be called on every batch'):
        scheduler.on_train_start(trainer, model)
    model = OverriddenZeroGradModel()
    with pytest.warns(UserWarning, match='the hooks will not be called on every batch'):
        scheduler.on_train_start(trainer, model)

def test_raises_when_accumulate_grad_batches_with_callback(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that it is not allowed to set both the Trainer argument and also pass a callback.'
    trainer = Trainer(default_root_dir=tmp_path, accumulate_grad_batches=2, callbacks=[GradientAccumulationScheduler({0: 2})])
    with pytest.raises(ValueError, match='`accumulate_grad_batches` and are using the `GradientAccumulationScheduler`'):
        trainer.fit(BoringModel())