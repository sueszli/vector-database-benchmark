from unittest import mock
from unittest.mock import Mock, call
import pytest
import torch
from lightning.fabric.accelerators.cuda import _clear_cuda_memory
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tests_pytorch.helpers.runif import RunIf

@mock.patch('lightning.pytorch.loops.evaluation_loop._EvaluationLoop._on_evaluation_epoch_end')
def test_on_evaluation_epoch_end(eval_epoch_end_mock, tmpdir):
    if False:
        while True:
            i = 10
    'Tests that `on_evaluation_epoch_end` is called for `on_validation_epoch_end` and `on_test_epoch_end` hooks.'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, enable_model_summary=False)
    trainer.fit(model)
    assert eval_epoch_end_mock.call_count == 3
    trainer.test()
    assert eval_epoch_end_mock.call_count == 4

@pytest.mark.parametrize('use_batch_sampler', [False, True])
def test_evaluation_loop_sampler_set_epoch_called(tmp_path, use_batch_sampler):
    if False:
        i = 10
        return i + 15
    "Tests that set_epoch is called on the dataloader's sampler (if any) during training and validation."

    def _get_dataloader():
        if False:
            print('Hello World!')
        dataset = RandomDataset(32, 64)
        sampler = RandomSampler(dataset)
        sampler.set_epoch = Mock()
        if use_batch_sampler:
            batch_sampler = BatchSampler(sampler, 2, True)
            return DataLoader(dataset, batch_sampler=batch_sampler)
        return DataLoader(dataset, sampler=sampler)
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=1, limit_val_batches=1, max_epochs=2, enable_model_summary=False, enable_checkpointing=False, logger=False)
    train_dataloader = _get_dataloader()
    val_dataloader = _get_dataloader()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    train_sampler = train_dataloader.batch_sampler.sampler if use_batch_sampler else train_dataloader.sampler
    val_sampler = val_dataloader.batch_sampler.sampler if use_batch_sampler else val_dataloader.sampler
    assert train_sampler.set_epoch.mock_calls == [call(0), call(1)]
    assert val_sampler.set_epoch.mock_calls == [call(0), call(0), call(1)]
    val_dataloader = _get_dataloader()
    trainer.validate(model, val_dataloader)
    val_sampler = val_dataloader.batch_sampler.sampler if use_batch_sampler else val_dataloader.sampler
    assert val_sampler.set_epoch.mock_calls == [call(2)]

@mock.patch('lightning.pytorch.trainer.connectors.logger_connector.logger_connector._LoggerConnector.log_eval_end_metrics')
def test_log_epoch_metrics_before_on_evaluation_end(update_eval_epoch_metrics_mock, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that the epoch metrics are logged before the `on_evaluation_end` hook is fired.'
    order = []
    update_eval_epoch_metrics_mock.side_effect = lambda _: order.append('log_epoch_metrics')

    class LessBoringModel(BoringModel):

        def on_validation_end(self):
            if False:
                while True:
                    i = 10
            order.append('on_validation_end')
            super().on_validation_end()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1, enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(LessBoringModel())
    assert order == ['log_epoch_metrics', 'on_validation_end']

@RunIf(min_cuda_gpus=1)
def test_memory_consumption_validation(tmpdir):
    if False:
        return 10
    'Test that the training batch is no longer in GPU memory when running validation.\n\n    Cannot run with MPS, since there we can only measure shared memory and not dedicated, which device has how much\n    memory allocated.\n\n    '

    def get_memory():
        if False:
            for i in range(10):
                print('nop')
        _clear_cuda_memory()
        return torch.cuda.memory_allocated(0)
    initial_memory = get_memory()

    class BoringLargeBatchModel(BoringModel):

        @property
        def num_params(self):
            if False:
                return 10
            return sum((p.numel() for p in self.parameters()))

        def train_dataloader(self):
            if False:
                for i in range(10):
                    print('nop')
            batch_size = self.num_params * 100 // 32 + 1
            return DataLoader(RandomDataset(32, 5000), batch_size=batch_size)

        def val_dataloader(self):
            if False:
                i = 10
                return i + 15
            return self.train_dataloader()

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            lower = 101 * self.num_params * 4
            upper = 201 * self.num_params * 4
            current = get_memory()
            assert lower < current
            assert current - initial_memory < upper
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            if False:
                return 10
            lower = 101 * self.num_params * 4
            upper = 201 * self.num_params * 4
            current = get_memory()
            assert lower < current
            assert current - initial_memory < upper
            return super().validation_step(batch, batch_idx)
    _clear_cuda_memory()
    trainer = Trainer(accelerator='gpu', devices=1, default_root_dir=tmpdir, fast_dev_run=2, enable_model_summary=False)
    trainer.fit(BoringLargeBatchModel())

def test_evaluation_loop_dataloader_iter_multiple_dataloaders(tmp_path):
    if False:
        print('Hello World!')
    trainer = Trainer(default_root_dir=tmp_path, limit_val_batches=1, enable_model_summary=False, enable_checkpointing=False, logger=False, devices=1)

    class MyModel(LightningModule):
        batch_start_ins = []
        step_outs = []
        batch_end_ins = []

        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            if False:
                i = 10
                return i + 15
            self.batch_start_ins.append((batch, batch_idx, dataloader_idx))

        def validation_step(self, dataloader_iter):
            if False:
                print('Hello World!')
            self.step_outs.append(next(dataloader_iter))

        def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            if False:
                i = 10
                return i + 15
            self.batch_end_ins.append((batch, batch_idx, dataloader_idx))
    model = MyModel()
    trainer.validate(model, {'a': [0, 1], 'b': [2, 3]})
    assert model.batch_start_ins == [(None, 0, 0), (0, 0, 0)]
    assert model.step_outs == [(0, 0, 0), (2, 0, 1)]
    assert model.batch_end_ins == model.step_outs

def test_invalid_dataloader_idx_raises_step(tmp_path):
    if False:
        return 10
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if False:
                i = 10
                return i + 15
            ...

        def test_step(self, batch, batch_idx, dataloader_idx):
            if False:
                i = 10
                return i + 15
            ...
    model = ExtraDataloaderIdx()
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.validation_step'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.test_step'):
        trainer.test(model)

    class GoodDefault(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                i = 10
                return i + 15
            ...

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                i = 10
                return i + 15
            ...
    model = GoodDefault()
    trainer.validate(model)
    trainer.test(model)

    class ExtraDlIdxOtherName(BoringModel):

        def validation_step(self, batch, batch_idx, dl_idx):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def test_step(self, batch, batch_idx, dl_idx):
            if False:
                print('Hello World!')
            ...
    model = ExtraDlIdxOtherName()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.validate(model)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.test(model)

    class MultipleDataloader(BoringModel):

        def val_dataloader(self):
            if False:
                return 10
            return [super().val_dataloader(), super().val_dataloader()]

        def test_dataloader(self):
            if False:
                return 10
            return [super().test_dataloader(), super().test_dataloader()]
    model = MultipleDataloader()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.validation_step'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.test_step'):
        trainer.test(model)

    class IgnoringModel(MultipleDataloader):

        def validation_step(self, batch, batch_idx, *_):
            if False:
                while True:
                    i = 10
            ...

        def test_step(self, batch, batch_idx, *_):
            if False:
                return 10
            ...
    model = IgnoringModel()
    trainer.validate(model)
    trainer.test(model)

    class IgnoringModel2(MultipleDataloader):

        def validation_step(self, batch, batch_idx, **_):
            if False:
                i = 10
                return i + 15
            ...

        def test_step(self, batch, batch_idx, **_):
            if False:
                print('Hello World!')
            ...
    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.validation_step'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.test_step'):
        trainer.test(model)

def test_invalid_dataloader_idx_raises_batch_start(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):

        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            if False:
                i = 10
                return i + 15
            ...

        def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
            if False:
                while True:
                    i = 10
            ...
    model = ExtraDataloaderIdx()
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.on_validation_batch_start'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.on_test_batch_start'):
        trainer.test(model)

    class GoodDefault(BoringModel):

        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
            if False:
                return 10
            ...

        def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
            if False:
                while True:
                    i = 10
            ...
    model = GoodDefault()
    trainer.validate(model)
    trainer.test(model)

    class ExtraDlIdxOtherName(BoringModel):

        def on_validation_batch_start(self, batch, batch_idx, dl_idx):
            if False:
                return 10
            ...

        def on_test_batch_start(self, batch, batch_idx, dl_idx):
            if False:
                i = 10
                return i + 15
            ...
    model = ExtraDlIdxOtherName()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.validate(model)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.test(model)

    class MultipleDataloader(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                i = 10
                return i + 15
            ...

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                while True:
                    i = 10
            ...

        def on_validation_batch_start(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def on_test_batch_start(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def val_dataloader(self):
            if False:
                print('Hello World!')
            return [super().val_dataloader(), super().val_dataloader()]

        def test_dataloader(self):
            if False:
                return 10
            return [super().test_dataloader(), super().test_dataloader()]
    model = MultipleDataloader()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.on_validation_batch_start'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.on_test_batch_start'):
        trainer.test(model)

    class IgnoringModel(MultipleDataloader):

        def on_validation_batch_start(self, batch, batch_idx, *_):
            if False:
                return 10
            ...

        def on_test_batch_start(self, batch, batch_idx, *_):
            if False:
                i = 10
                return i + 15
            ...
    model = IgnoringModel()
    trainer.validate(model)
    trainer.test(model)

    class IgnoringModel2(MultipleDataloader):

        def on_validation_batch_start(self, batch, batch_idx, **_):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def on_test_batch_start(self, batch, batch_idx, **_):
            if False:
                while True:
                    i = 10
            ...
    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.on_validation_batch_start'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.on_test_batch_start'):
        trainer.test(model)

def test_invalid_dataloader_idx_raises_batch_end(tmp_path):
    if False:
        i = 10
        return i + 15
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):

        def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            if False:
                return 10
            ...
    model = ExtraDataloaderIdx()
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.on_validation_batch_end'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='have included `dataloader_idx` in `ExtraDataloaderIdx.on_test_batch_end'):
        trainer.test(model)

    class GoodDefault(BoringModel):

        def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
            if False:
                return 10
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
            if False:
                i = 10
                return i + 15
            ...
    model = GoodDefault()
    trainer.validate(model)
    trainer.test(model)

    class ExtraDlIdxOtherName(BoringModel):

        def on_validation_batch_end(self, outputs, batch, batch_idx, dl_idx):
            if False:
                i = 10
                return i + 15
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx, dl_idx):
            if False:
                i = 10
                return i + 15
            ...
    model = ExtraDlIdxOtherName()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.validate(model)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.test(model)

    class MultipleDataloader(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                while True:
                    i = 10
            ...

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                i = 10
                return i + 15
            ...

        def on_validation_batch_end(self, outputs, batch, batch_idx):
            if False:
                i = 10
                return i + 15
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx):
            if False:
                i = 10
                return i + 15
            ...

        def val_dataloader(self):
            if False:
                print('Hello World!')
            return [super().val_dataloader(), super().val_dataloader()]

        def test_dataloader(self):
            if False:
                for i in range(10):
                    print('nop')
            return [super().test_dataloader(), super().test_dataloader()]
    model = MultipleDataloader()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.on_validation_batch_end'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `MultipleDataloader.on_test_batch_end'):
        trainer.test(model)

    class IgnoringModel(MultipleDataloader):

        def on_validation_batch_end(self, outputs, batch, batch_idx, *_):
            if False:
                print('Hello World!')
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx, *_):
            if False:
                return 10
            ...
    model = IgnoringModel()
    trainer.validate(model)
    trainer.test(model)

    class IgnoringModel2(MultipleDataloader):

        def on_validation_batch_end(self, outputs, batch, batch_idx, **_):
            if False:
                for i in range(10):
                    print('nop')
            ...

        def on_test_batch_end(self, outputs, batch, batch_idx, **_):
            if False:
                i = 10
                return i + 15
            ...
    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.on_validation_batch_end'):
        trainer.validate(model)
    with pytest.raises(RuntimeError, match='no `dataloader_idx` argument in `IgnoringModel2.on_test_batch_end'):
        trainer.test(model)

@pytest.mark.parametrize(('mode', 'expected'), [('max_size_cycle', [{'a': 0, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}]), ('min_size', [{'a': 0, 'b': 3}, {'a': 1, 'b': 4}]), ('max_size', [{'a': 0, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': None}])])
@pytest.mark.parametrize('fn', ['validate', 'test'])
def test_evaluation_loop_non_sequential_mode_supprt(tmp_path, mode, expected, fn):
    if False:
        while True:
            i = 10
    iterables = {'a': [0, 1, 2], 'b': {3, 4}}
    cl = CombinedLoader(iterables, mode)
    seen = []

    class MyModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            seen.append(batch)

        def test_step(self, batch, batch_idx):
            if False:
                return 10
            seen.append(batch)
    model = MyModel()
    trainer = Trainer(default_root_dir=tmp_path, barebones=True)
    trainer_fn = getattr(trainer, fn)
    trainer_fn(model, cl)
    assert trainer.num_sanity_val_batches == []
    actual = trainer.num_val_batches if fn == 'validate' else trainer.num_test_batches
    assert actual == [3, 2]
    assert seen == expected

def test_evaluation_loop_when_batch_idx_argument_is_not_given(tmpdir):
    if False:
        i = 10
        return i + 15

    class TestModel(BoringModel):

        def __init__(self) -> None:
            if False:
                while True:
                    i = 10
            super().__init__()
            self.validation_step_called = False
            self.test_step_called = False

        def validation_step(self, batch):
            if False:
                while True:
                    i = 10
            self.validation_step_called = True
            return {'x': self.step(batch)}

        def test_step(self, batch):
            if False:
                for i in range(10):
                    print('nop')
            self.test_step_called = True
            return {'y': self.step(batch)}
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    model = TestModel()
    trainer.validate(model)
    assert model.validation_step_called
    trainer.test(model)
    assert model.test_step_called