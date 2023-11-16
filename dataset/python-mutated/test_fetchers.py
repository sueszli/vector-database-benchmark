from collections import Counter
from typing import Any, Iterator
import pytest
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loops.fetchers import _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tests_pytorch.helpers.runif import RunIf

class IterDataset(IterableDataset):

    def __init__(self, size=3):
        if False:
            return 10
        self.size = size

    def __iter__(self):
        if False:
            print('Hello World!')
        yield from range(1, self.size + 1)

class SizedDataset(Dataset):

    def __len__(self):
        if False:
            print('Hello World!')
        return 3

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        return idx + 1

@pytest.mark.parametrize('multiple_iterables', [False, True])
@pytest.mark.parametrize('dataset_cls', [IterDataset, SizedDataset])
@pytest.mark.parametrize('prefetch_batches', list(range(5)))
def test_prefetch_iterator(multiple_iterables, dataset_cls, prefetch_batches):
    if False:
        while True:
            i = 10
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    assert fetcher.prefetch_batches == prefetch_batches
    if multiple_iterables:
        loader = CombinedLoader([DataLoader(dataset_cls()), DataLoader(dataset_cls())])
    else:
        loader = CombinedLoader(DataLoader(dataset_cls()))
    fetcher.setup(loader)

    def generate():
        if False:
            while True:
                i = 10
        generated = [(fetcher.fetched, data, fetcher.done) for (data, batch_idx, dataloader_idx) in fetcher]
        assert fetcher.fetched == 3
        assert fetcher.done
        return generated
    is_last_batch = [False, False, prefetch_batches > 0 or dataset_cls is SizedDataset]
    fetched = [1, 2, 3] if dataset_cls is SizedDataset else [1, 2, 3, 3, 3, 3, 3][prefetch_batches:prefetch_batches + 3]
    batches = [[1, 1], [2, 2], [3, 3]] if multiple_iterables else [1, 2, 3]
    expected = list(zip(fetched, batches, is_last_batch))
    assert len(expected) == 3
    assert generate() == expected
    assert generate() == expected
    assert fetcher.fetched == 3

@pytest.mark.parametrize('multiple_iterables', [False, True])
def test_profiler_closing(multiple_iterables):
    if False:
        while True:
            i = 10
    'Tests if the profiler terminates upon raising a StopIteration on an iterable dataset.'

    class TestDataset(IterableDataset):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.list = list(range(1))

        def __iter__(self):
            if False:
                print('Hello World!')
            return iter(self.list)
    fetcher = _PrefetchDataFetcher()
    if multiple_iterables:
        loader = CombinedLoader([DataLoader(TestDataset()), DataLoader(TestDataset())])
    else:
        loader = CombinedLoader(TestDataset())
    fetcher.setup(loader)
    profiler = SimpleProfiler()
    fetcher._start_profiler = lambda : profiler.start('test')
    fetcher._stop_profiler = lambda : profiler.stop('test')
    iter(fetcher)
    next(fetcher)
    assert not bool(profiler.current_actions)

class EmptyIterDataset(IterableDataset):

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter([])

class EmptySizedDataset(Dataset):

    def __len__(self):
        if False:
            print('Hello World!')
        return 0

@pytest.mark.parametrize('dataset_cls', [EmptyIterDataset, EmptySizedDataset])
@pytest.mark.parametrize('prefetch_batches', [0, 1])
def test_empty_prefetch_iterator(dataset_cls, prefetch_batches):
    if False:
        while True:
            i = 10
    loader = CombinedLoader(DataLoader(dataset_cls()))
    fetcher = _PrefetchDataFetcher(prefetch_batches=prefetch_batches)
    fetcher.setup(loader)
    iter(fetcher)
    if dataset_cls is EmptySizedDataset:
        assert fetcher.done
    else:
        assert fetcher.done == (prefetch_batches > 0)
    assert not list(fetcher)
    assert fetcher.done

def get_cycles_per_ms() -> float:
    if False:
        while True:
            i = 10
    'Get 10 values and remove the 2 max and 2 min and return the avg.\n\n    This is to avoid system disturbance that skew the results, e.g. the very first cuda call likely does a bunch of\n    init, which takes much longer than subsequent calls.\n\n    '

    def measure() -> float:
        if False:
            while True:
                i = 10
        'Measure and return approximate number of cycles per millisecond for `torch.cuda._sleep` Copied from:\n\n        https://github.com/pytorch/pytorch/blob/v1.9.0/test/test_cuda.py#L81.\n\n        '
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        return 1000000 / start.elapsed_time(end)
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    stats = vals[2:num - 2]
    return sum(stats) / len(stats)
BATCH_SIZE = 32
DATASET_LEN = 64

@pytest.mark.parametrize('automatic_optimization', [False, True])
def test_fetching_dataloader_iter_opt(automatic_optimization, tmpdir):
    if False:
        i = 10
        return i + 15

    class TestModel(BoringModel):

        def __init__(self, *args, automatic_optimization: bool=False, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self.automatic_optimization = automatic_optimization
            self.count = 0
            self.batches = []

        def training_step(self, dataloader_iter):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(self.trainer.fit_loop._data_fetcher, _DataLoaderIterDataFetcher)
            (batch, batch_idx, dataloader_idx) = next(dataloader_iter)
            self.batches.append(batch)
            (batch, batch_idx, dataloader_idx) = next(dataloader_iter)
            self.batches.append(batch)
            batch = self.batches.pop(0)
            assert isinstance(batch, Tensor) or batch is None
            self.count = batch_idx + 1
            if self.automatic_optimization:
                loss = super().training_step(batch, 0)
                with pytest.raises(MisconfigurationException, match='dataloader_iter'):
                    self.log('train_loss', loss['loss'])
                self.log('train_loss', loss['loss'], batch_size=1)
            else:
                opt = self.optimizers()
                loss = self.step(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        def on_train_epoch_end(self):
            if False:
                while True:
                    i = 10
            assert self.trainer.fit_loop.epoch_loop.batch_progress.current.ready == 32
            assert self.trainer.fit_loop._data_fetcher.fetched == 64
            assert self.count == 64
    model = TestModel(automatic_optimization=automatic_optimization)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator='cpu')
    trainer.fit(model)

@pytest.mark.parametrize('fn', ['validate', 'test', 'predict'])
def test_fetching_dataloader_iter_running_stages(fn, tmp_path):
    if False:
        print('Hello World!')

    class TestModel(BoringModel):

        def fetch(self, data_fetcher, dataloader_iter):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(data_fetcher, _DataLoaderIterDataFetcher)
            (batch, batch_idx, dataloader_idx) = next(dataloader_iter)
            assert data_fetcher.fetched == batch_idx + 1
            return batch

        def validation_step(self, dataloader_iter):
            if False:
                print('Hello World!')
            data_fetcher = self.trainer.validate_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().validation_step(batch, 0)

        def test_step(self, dataloader_iter):
            if False:
                for i in range(10):
                    print('nop')
            data_fetcher = self.trainer.test_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().test_step(batch, 0)

        def predict_step(self, dataloader_iter):
            if False:
                for i in range(10):
                    print('nop')
            data_fetcher = self.trainer.predict_loop._data_fetcher
            batch = self.fetch(data_fetcher, dataloader_iter)
            return super().test_step(batch, 0)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1, accelerator='cpu')
    trainer_fn = getattr(trainer, fn)
    trainer_fn(model)

class DummyWaitable:

    def __init__(self, val: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.val = val

    def wait(self) -> Any:
        if False:
            print('Hello World!')
        return self.val

class AsyncBoringModel(BoringModel):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.automatic_optimization = False
        self.batch_i_handle = None
        self.num_batches_processed = 0

    def _async_op(self, batch: Any) -> DummyWaitable:
        if False:
            i = 10
            return i + 15
        return DummyWaitable(val=batch)

    def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
        if False:
            while True:
                i = 10
        if self.batch_i_handle is None:
            (batch_i_raw, _, _) = next(dataloader_iter)
            self.num_batches_processed += 1
            self.batch_i_handle = self._async_op(batch_i_raw)
        batch_ip1_handle = None
        is_last = False
        try:
            (batch_ip1_raw, _, _) = next(dataloader_iter)
            self.num_batches_processed += 1
            batch_ip1_handle = self._async_op(batch_ip1_raw)
        except StopIteration:
            is_last = True
        batch_i = self.batch_i_handle.wait()
        loss = self.step(batch_i)
        loss.backward()
        self.optimizers().step()
        self.optimizers().zero_grad()
        self.batch_i_handle = batch_ip1_handle
        return {'loss': loss, 'is_last': is_last}

    def train_dataloader(self):
        if False:
            for i in range(10):
                print('nop')
        return DataLoader(RandomDataset(BATCH_SIZE, DATASET_LEN))

def test_training_step_with_dataloader_iter(tmpdir) -> None:
    if False:
        for i in range(10):
            print('nop')
    'A baseline functional test for `training_step` with dataloader access.'
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, accelerator='cpu')
    m = AsyncBoringModel()
    trainer.fit(m)
    assert m.num_batches_processed == DATASET_LEN, f'Expect all {DATASET_LEN} batches to be processed.'

class DataLoaderIterMonitorModel(BoringModel):

    def __init__(self, fetches_per_step):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fetches_per_step = fetches_per_step
        self.record = {'training': Counter(), 'validation': Counter(), 'sanity_validation': Counter(), 'test': Counter(), 'predict': Counter()}

    def shared_step(self, dataloader_iter, stage):
        if False:
            while True:
                i = 10
        self.record[stage]['entered'] += 1
        for i in range(self.fetches_per_step):
            try:
                (batch, _, __) = next(dataloader_iter)
            except StopIteration:
                self.record[stage]['raised'] += 1
                return None
            self.record[stage]['fetched'] += 1
        return self.layer(batch).sum()

    def training_step(self, dataloader_iter):
        if False:
            while True:
                i = 10
        return self.shared_step(dataloader_iter, 'training')

    def validation_step(self, dataloader_iter):
        if False:
            for i in range(10):
                print('nop')
        stage = 'sanity_validation' if self.trainer.sanity_checking else 'validation'
        return self.shared_step(dataloader_iter, stage)

    def test_step(self, dataloader_iter):
        if False:
            print('Hello World!')
        return self.shared_step(dataloader_iter, 'test')

    def predict_step(self, dataloader_iter):
        if False:
            while True:
                i = 10
        return self.shared_step(dataloader_iter, 'predict')

@pytest.mark.parametrize(('limit_sanity_val_batches', 'limit_train_batches', 'limit_eval_batches'), [(None, None, None), (0, 0, 0), (2, 2, 2), (100, 100, 100)])
def test_step_methods_with_dataloader_iter(limit_sanity_val_batches, limit_train_batches, limit_eval_batches, tmp_path):
    if False:
        return 10
    global_batch_size = 4
    micro_batch_size = 2
    fetches_per_step = global_batch_size // micro_batch_size
    data = DataLoader(RandomDataset(32, length=16), batch_size=micro_batch_size)
    assert len(data) == 8
    limit_sanity_val_batches = 2 if limit_sanity_val_batches is None else limit_sanity_val_batches
    limit_train_batches = limit_train_batches
    limit_val_batches = limit_eval_batches
    limit_test_batches = limit_eval_batches
    limit_predict_batches = limit_eval_batches
    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, limit_test_batches=limit_test_batches, limit_predict_batches=limit_predict_batches, num_sanity_val_steps=limit_sanity_val_batches, max_epochs=1, accelerator='cpu', logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, data, data)

    def length(iterable, limit):
        if False:
            while True:
                i = 10
        return len(iterable) if limit is None else min(limit, len(data))
    assert model.record['sanity_validation']['entered'] == length(data, limit_sanity_val_batches) // fetches_per_step
    assert model.record['sanity_validation']['fetched'] == length(data, limit_sanity_val_batches)
    assert model.record['sanity_validation']['raised'] == 0
    assert model.record['training']['entered'] == length(data, limit_train_batches) // fetches_per_step
    assert model.record['training']['fetched'] == length(data, limit_train_batches)
    assert model.record['training']['raised'] == 0
    assert model.record['validation']['entered'] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record['validation']['fetched'] == length(data, limit_eval_batches)
    assert model.record['validation']['raised'] == 0
    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.validate(model, data)
    assert model.record['validation']['entered'] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record['validation']['fetched'] == length(data, limit_eval_batches)
    assert model.record['validation']['raised'] == 0
    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.test(model, data)
    assert model.record['test']['entered'] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record['test']['fetched'] == length(data, limit_eval_batches)
    assert model.record['test']['raised'] == 0
    model = DataLoaderIterMonitorModel(fetches_per_step)
    trainer.predict(model, data)
    assert model.record['predict']['entered'] == length(data, limit_eval_batches) // fetches_per_step
    assert model.record['predict']['fetched'] == length(data, limit_eval_batches)
    assert model.record['predict']['raised'] == 0

@pytest.mark.parametrize('trigger_stop_iteration', [False, True])
def test_stop_iteration_with_dataloader_iter(trigger_stop_iteration, tmpdir):
    if False:
        print('Hello World!')
    'Verify that StopIteration properly terminates the training when this is triggered from the current\n    `dataloader_iter`'
    EXPECT_NUM_BATCHES_PROCESSED = 2

    class TestModel(AsyncBoringModel):

        def __init__(self, trigger_stop_iteration) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.trigger_stop_iteration = trigger_stop_iteration

        def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:
            if False:
                print('Hello World!')
            output = super().training_step(dataloader_iter)
            batch_idx = self.trainer.fit_loop.epoch_loop.batch_idx
            if self.trigger_stop_iteration and batch_idx == EXPECT_NUM_BATCHES_PROCESSED:
                raise StopIteration
            return output

        def train_dataloader(self):
            if False:
                return 10
            if self.trigger_stop_iteration:
                return DataLoader(RandomDataset(BATCH_SIZE, 2 * EXPECT_NUM_BATCHES_PROCESSED))
            return DataLoader(RandomDataset(BATCH_SIZE, EXPECT_NUM_BATCHES_PROCESSED))
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, accelerator='cpu')
    m = TestModel(trigger_stop_iteration)
    trainer.fit(m)
    expected = EXPECT_NUM_BATCHES_PROCESSED
    if trigger_stop_iteration:
        expected *= 2
    assert m.num_batches_processed == expected

def test_transfer_hooks_with_unpacking(tmpdir):
    if False:
        print('Hello World!')
    'This test asserts the `transfer_batch` hooks are called only once per batch.'

    class RandomDictDataset(RandomDataset):

        def __getitem__(self, index):
            if False:
                for i in range(10):
                    print('nop')
            return {'x': self.data[index], 'y_true': torch.ones((2,)), 'other': torch.ones((1,))}

    class BoringDataModule(LightningDataModule):
        count_called_on_before_batch_transfer = 0
        count_called_transfer_batch_to_device = 0
        count_called_on_after_batch_transfer = 0

        def train_dataloader(self):
            if False:
                return 10
            return DataLoader(RandomDictDataset(32, 2))

        def val_dataloader(self):
            if False:
                while True:
                    i = 10
            return DataLoader(RandomDictDataset(32, 2))

        def on_before_batch_transfer(self, batch, dataloader_idx: int):
            if False:
                return 10
            self.count_called_on_before_batch_transfer += 1
            return (batch['x'], batch['y_true'])

        def transfer_batch_to_device(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.count_called_transfer_batch_to_device += 1
            return super().transfer_batch_to_device(*args, **kwargs)

        def on_after_batch_transfer(self, batch, dataloader_idx: int):
            if False:
                while True:
                    i = 10
            self.count_called_on_after_batch_transfer += 1
            return super().on_after_batch_transfer(batch, dataloader_idx)

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            (x, _) = batch
            return super().training_step(x, batch_idx)

        def validation_step(self, batch, batch_idx):
            if False:
                while True:
                    i = 10
            (x, _) = batch
            return super().validation_step(x, batch_idx)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, num_sanity_val_steps=0)
    dm = BoringDataModule()
    trainer.fit(TestModel(), datamodule=dm)
    assert dm.count_called_on_before_batch_transfer == 4
    assert dm.count_called_transfer_batch_to_device == 4
    assert dm.count_called_on_after_batch_transfer == 4

@RunIf(skip_windows=True)
def test_fetching_is_profiled():
    if False:
        print('Hello World!')
    'Test that fetching is profiled.'

    class MyModel(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if False:
                while True:
                    i = 10
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            if False:
                while True:
                    i = 10
            return [super().val_dataloader(), super().val_dataloader()]
    model = MyModel()
    fast_dev_run = 2
    trainer = Trainer(fast_dev_run=fast_dev_run, profiler='simple', enable_model_summary=False, enable_checkpointing=False, enable_progress_bar=False, logger=False, accelerator='cpu')
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)
    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)
    key = '[_EvaluationLoop].val_next'
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == 2 * fast_dev_run
    assert all((d > 0 for d in durations))
    key = '[_TrainingEpochLoop].train_dataloader_next'
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run
    assert all((d > 0 for d in durations))
    key = '[_EvaluationLoop].test_next'
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run
    assert all((d > 0 for d in durations))
    key = '[_PredictionLoop].predict_next'
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == fast_dev_run
    assert all((d > 0 for d in durations))

    class MyModel(BoringModel):

        def training_step(self, dataloader_iter):
            if False:
                return 10
            _ = next(dataloader_iter)
            (batch, _, _) = next(dataloader_iter)
            return super().training_step(batch, 0)
    model = MyModel()
    trainer = Trainer(fast_dev_run=2, profiler='simple', limit_val_batches=0, enable_model_summary=False, enable_checkpointing=False, enable_progress_bar=False, logger=False, accelerator='cpu')
    trainer.fit(model)
    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)
    key = '[_TrainingEpochLoop].train_dataloader_next'
    assert key in profiler.recorded_durations
    durations = profiler.recorded_durations[key]
    assert len(durations) == 2
    assert all((d > 0 for d in durations))

@pytest.mark.parametrize('iterable', [[1, 2, 3], IterDataset()])
def test_done_dataloader_iter(iterable):
    if False:
        return 10
    loader = CombinedLoader(iterable)
    fetcher = _DataLoaderIterDataFetcher()
    fetcher.setup(loader)
    iter(fetcher)
    assert not fetcher.done
    dataloader_iter = next(fetcher)
    for i in range(5):
        assert next(fetcher) is next(fetcher)
    assert not dataloader_iter.done
    assert dataloader_iter.data_fetcher is fetcher
    assert not dataloader_iter.done
    assert next(dataloader_iter)[0] == 1
    assert not dataloader_iter.done
    assert next(dataloader_iter)[0] == 2
    assert not dataloader_iter.done
    assert next(dataloader_iter)[0] == 3
    if isinstance(iterable, list):
        assert dataloader_iter.done
    else:
        assert not dataloader_iter.done
    with pytest.raises(StopIteration):
        next(dataloader_iter)
    assert dataloader_iter.done

@pytest.mark.parametrize(('mode', 'iterables', 'limit', 'num_fetches', 'expected'), [('min_size', [[1, 2, 3]], None, 2, False), ('min_size', [[1, 2, 3]], None, 3, True), ('min_size', [[1, 2, 3]], 1, 1, True), ('min_size', [[1, 2], [1, 2, 3]], None, 1, False), ('min_size', [[1, 2], [1, 2, 3]], None, 2, True), ('min_size', [[1, 2], [1, 2, 3]], 1, 1, True), ('max_size', [[1, 2], [1, 2, 3]], None, 2, False), ('max_size', [[1, 2], [1, 2, 3]], 2, 2, True), ('max_size', [[1, 2], [1, 2, 3]], 100, 3, True), ('max_size_cycle', [[1, 2], [1, 2, 3]], None, 2, False), ('max_size_cycle', [[1, 2], [1, 2, 3]], 2, 2, True), ('max_size_cycle', [[1, 2], [1, 2, 3]], 100, 3, True), ('sequential', [[1, 2], [1, 2, 3]], None, 2, False), ('sequential', [[1, 2], [1, 2, 3]], 2, 2, False), ('sequential', [[1, 2], [1, 2, 3]], 2, 4, True), ('sequential', [[1, 2], [1, 2, 3]], 100, 5, True), ('min_size', [IterDataset()], None, 2, False), ('min_size', [IterDataset()], None, 3, False), ('min_size', [IterDataset()], 1, 1, True), ('min_size', [IterDataset(2), IterDataset(3)], None, 1, False), ('min_size', [IterDataset(2), IterDataset(3)], None, 2, False), ('min_size', [IterDataset(2), IterDataset(3)], 1, 1, True), ('max_size', [IterDataset(2), IterDataset(3)], None, 2, False), ('max_size', [IterDataset(2), IterDataset(3)], 2, 2, True), ('max_size', [IterDataset(2), IterDataset(3)], 100, 3, False), ('max_size_cycle', [IterDataset(2), IterDataset(3)], None, 2, False), ('max_size_cycle', [IterDataset(2), IterDataset(3)], 2, 2, True), ('max_size_cycle', [IterDataset(2), IterDataset(3)], 100, 3, False), ('sequential', [IterDataset(2), IterDataset(3)], None, 2, False), ('sequential', [IterDataset(2), IterDataset(3)], 2, 2, False), ('sequential', [IterDataset(2), IterDataset(3)], 2, 4, True), ('sequential', [IterDataset(2), IterDataset(3)], 100, 5, False), ('min_size', [[1, 2], IterDataset(3)], None, 1, False), ('min_size', [[1, 2], IterDataset(3)], None, 2, True), ('max_size', [IterDataset(2), [1, 2, 3]], None, 2, False), ('max_size', [IterDataset(2), [1, 2, 3]], None, 3, False), ('max_size_cycle', [IterDataset(2), [1, 2, 3]], None, 2, False), ('max_size_cycle', [IterDataset(2), [1, 2, 3]], None, 3, False), ('sequential', [[1, 2], IterDataset(3)], 2, 2, False), ('sequential', [[1, 2], IterDataset(3)], 2, 4, True)])
def test_done_dataloader_iter_with_limit(mode, iterables, limit, num_fetches, expected):
    if False:
        while True:
            i = 10
    'Test that the `done` property for `dataloader_iter` gets set as expected.'
    loader = CombinedLoader(iterables, mode=mode)
    fetcher = _DataLoaderIterDataFetcher()
    loader.limits = limit
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done == (limit == 0)
    if num_fetches == 0:
        return
    dataloader_iter = next(fetcher)
    assert not dataloader_iter.done
    for _ in range(num_fetches):
        next(dataloader_iter)
    assert dataloader_iter.done == expected
    assert fetcher.done == expected
    if fetcher.done:
        with pytest.raises(StopIteration):
            next(dataloader_iter)

@pytest.mark.parametrize('mode', ['min_size', 'max_size_cycle', 'max_size', 'sequential'])
def test_done_dataloader_iter_empty_iterables(mode):
    if False:
        return 10
    'Test that the `done` property for `dataloader_iter` gets set as expected for empty iterables.'
    fetcher = _DataLoaderIterDataFetcher()
    loader = CombinedLoader([], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done
    loader = CombinedLoader([[], []], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done
    loader = CombinedLoader([[], [1, 2, 3]], mode=mode)
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done == (mode == 'min_size')

@pytest.mark.parametrize('mode', ['min_size', 'max_size_cycle', 'max_size', 'sequential'])
@pytest.mark.parametrize('iterables', [[], [IterDataset()], [[], [1, 2, 3]]])
def test_done_dataloader_iter_zero_limit(iterables, mode):
    if False:
        return 10
    'Test that the `done` property for `dataloader_iter` gets set as expected when the limit is 0.'
    fetcher = _DataLoaderIterDataFetcher()
    loader = CombinedLoader(iterables, mode=mode)
    loader.limits = 0
    fetcher.setup(loader)
    iter(fetcher)
    assert fetcher.done