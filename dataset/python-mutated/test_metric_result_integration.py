import os
import pickle
from contextlib import nullcontext, suppress
from unittest import mock
import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import OnExceptionCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.connectors.logger_connector.result import _Metadata, _ResultCollection, _ResultMetric, _Sync
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_11 as _TM_GE_0_11
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor, tensor
from torch.nn import ModuleDict, ModuleList
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy
from tests_pytorch.core.test_results import spawn_launch
from tests_pytorch.helpers.runif import RunIf

class DummyMetric(Metric):
    x: Tensor

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.add_state('x', tensor(0), dist_reduce_fx='sum')

    def update(self, x):
        if False:
            i = 10
            return i + 15
        self.x += x

    def compute(self):
        if False:
            print('Hello World!')
        return self.x

def result_reduce_ddp_fn(strategy):
    if False:
        return 10
    rank = strategy.local_rank
    worldsize = strategy.num_processes
    tensor([1.0])
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()
    metric_a = metric_a.to(f'cuda:{rank}')
    metric_b = metric_b.to(f'cuda:{rank}')
    metric_c = metric_c.to(f'cuda:{rank}')
    result = _ResultCollection(True)
    for _ in range(3):
        cumulative_sum = 0
        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)
            cumulative_sum += i
            result.log('h', 'a', metric_a, on_step=True, on_epoch=True)
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True)
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False)
            batch_log = result.metrics(True)['log']
            assert batch_log == {'a_step': i, 'c': i}
        epoch_log = result.metrics(False)['log']
        result.reset()
        assert metric_a.x == metric_a._defaults['x'], (metric_a.x, metric_a._defaults['x'])
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']
        assert epoch_log == {'b': cumulative_sum * worldsize, 'a_epoch': cumulative_sum * worldsize}

@RunIf(min_cuda_gpus=2, skip_windows=True)
def test_result_reduce_ddp():
    if False:
        while True:
            i = 10
    'Make sure result logging works with DDP.'
    spawn_launch(result_reduce_ddp_fn, [torch.device('cuda:0'), torch.device('cuda:1')])

def test_result_metric_integration():
    if False:
        print('Hello World!')
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()
    result = _ResultCollection(True)
    for _ in range(3):
        cumulative_sum = 0
        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)
            cumulative_sum += i
            result.log('h', 'a', metric_a, on_step=True, on_epoch=True)
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True)
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False)
            batch_log = result.metrics(True)['log']
            assert batch_log == {'a_step': i, 'c': i}
        epoch_log = result.metrics(False)['log']
        result.reset()
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']
        assert epoch_log == {'b': cumulative_sum, 'a_epoch': cumulative_sum}
    result.minimize = tensor(1.0)
    result.extra = {}
    assert str(result) == "_ResultCollection({'h.a': _ResultMetric('a', value=DummyMetric()), 'h.b': _ResultMetric('b', value=DummyMetric()), 'h.c': _ResultMetric('c', value=DummyMetric())})"
    assert repr(result) == "{True, {'h.a': _ResultMetric('a', value=DummyMetric()), 'h.b': _ResultMetric('b', value=DummyMetric()), 'h.c': _ResultMetric('c', value=DummyMetric())}}"

def test_result_collection_simple_loop():
    if False:
        print('Hello World!')
    result = _ResultCollection(True)
    current_fx_name = None
    batch_idx = None

    def lightning_log(fx, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        nonlocal current_fx_name
        if current_fx_name != fx and batch_idx in (None, 0):
            result.reset(metrics=False, fx=fx)
        result.log(fx, *args, **kwargs)
        current_fx_name = fx
    lightning_log('a0', 'a', tensor(0.0), on_step=True, on_epoch=True)
    lightning_log('a1', 'a', tensor(0.0), on_step=True, on_epoch=True)
    for epoch in range(2):
        lightning_log('b0', 'a', tensor(1.0) + epoch, on_step=True, on_epoch=True)
        lightning_log('b1', 'a', tensor(1.0) + epoch, on_step=True, on_epoch=True)
        for batch_idx in range(2):
            lightning_log('c0', 'a', tensor(2.0) + epoch, on_step=True, on_epoch=True)
            lightning_log('c1', 'a', tensor(2.0) + epoch, on_step=True, on_epoch=True)
            lightning_log('c2', 'a', tensor(2.0) + epoch, on_step=True, on_epoch=True)
        batch_idx = None
        lightning_log('d0', 'a', tensor(3.0) + epoch, on_step=False, on_epoch=True)
        lightning_log('d1', 'a', tensor(3.0) + epoch, on_step=False, on_epoch=True)
        for k in ('a0.a', 'a1.a'):
            assert result[k].value == tensor(0.0), k
            assert result[k].cumulated_batch_size == tensor(1.0), k
        for k in ('b0.a', 'b1.a'):
            assert result[k].value == tensor(1.0) + epoch, k
            assert result[k].cumulated_batch_size == tensor(1.0), k
        for k in ('c0.a', 'c1.a', 'c2.a'):
            assert result[k].value == tensor(4.0) + epoch * 2, k
            assert result[k].cumulated_batch_size == tensor(2.0), k
        for k in ('d0.a', 'd1.a'):
            assert result[k].value == tensor(3.0) + epoch, k
            assert result[k].cumulated_batch_size == tensor(1.0), k

def my_sync_dist(x, *_, **__):
    if False:
        while True:
            i = 10
    return x

def test_result_collection_restoration(tmpdir):
    if False:
        i = 10
        return i + 15
    'This test make sure metrics are properly reloaded on failure.'
    result = _ResultCollection(True)
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()
    metric_d = DummyMetric()
    current_fx_name = None
    batch_idx = None

    def lightning_log(fx, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        nonlocal current_fx_name
        if current_fx_name != fx and batch_idx in (None, 0):
            result.reset(metrics=False, fx=fx)
        result.log(fx, *args, **kwargs, sync_dist_fn=my_sync_dist)
        current_fx_name = fx
    for epoch in range(2):
        cumulative_sum = 0
        for i in range(3):
            a = metric_a(i)
            b = metric_b(i)
            c = metric_c(i)
            metric_d(i)
            cumulative_sum += i
            metric = metric_a if i < 1 else metric_d
            lightning_log('training_step', 'a', metric, on_step=True, on_epoch=True, metric_attribute='metric')
            lightning_log('training_step', 'b', metric_b, on_step=False, on_epoch=True, metric_attribute='metric_b')
            lightning_log('training_step', 'c', metric_c, on_step=True, on_epoch=False, metric_attribute='metric_c')
            lightning_log('training_step', 'a_1', a, on_step=True, on_epoch=True)
            lightning_log('training_step', 'b_1', b, on_step=False, on_epoch=True)
            lightning_log('training_step', 'c_1', c, on_step=True, on_epoch=False)
            batch_log = result.metrics(on_step=True)['log']
            assert set(batch_log) == {'a_step', 'c', 'a_1_step', 'c_1'}
            assert len(result.result_metrics) == 6 + epoch > 0
        lightning_log('train_epoch_end', 'a', metric_a, on_step=False, on_epoch=True)
        epoch_log = result.metrics(on_step=False)['log']
        assert epoch_log == {'a_1_epoch': 1, 'a_epoch': cumulative_sum, 'a': cumulative_sum, 'b': cumulative_sum, 'b_1': 1}
        pickle.loads(pickle.dumps(result))
        filepath = str(tmpdir / 'result')
        torch.save(result, filepath)
        torch.load(filepath)
        result.reset()
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']
        batch_idx = None

class DummyMeanMetric(Metric):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.add_state('sum', tensor(0), dist_reduce_fx=torch.sum)
        self.add_state('count', tensor(0), dist_reduce_fx=torch.sum)

    def update(self, increment):
        if False:
            i = 10
            return i + 15
        self.sum += increment
        self.count += 1

    def compute(self):
        if False:
            i = 10
            return i + 15
        return self.sum // self.count

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(sum={self.sum}, count={self.count})'

def result_collection_reload(default_root_dir, accelerator='auto', devices=1, **kwargs):
    if False:
        print('Hello World!')

    class CustomException(Exception):
        pass
    batches = 5

    class ExtendedBoringModel(BoringModel):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.breaking_batch_idx = 3
            self.has_validated_sum = False
            self.dummy_metric = DummyMeanMetric()

        @property
        def results(self):
            if False:
                return 10
            return self.trainer.fit_loop._results

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            if self.trainer.fit_loop.restarting:
                self.log('tracking', batch_idx, on_step=True, on_epoch=True)
                self.log('tracking_2', batch_idx, on_step=True, on_epoch=True, sync_dist=True)
                self.dummy_metric(batch_idx)
                self.log('tracking_metric', self.dummy_metric, on_step=True, on_epoch=True)
                value = self.results['training_step.tracking_metric']
                value_2 = self.results['training_step.tracking']
                shift = 0
                if devices == 2:
                    shift = 3 if self.trainer.is_global_zero else -3
                expected = sum(range(batch_idx + 1)) + shift
                assert expected == value == value_2
            else:
                if batch_idx == self.breaking_batch_idx:
                    raise CustomException
                self.log('tracking', batch_idx, on_step=True, on_epoch=True)
                self.log('tracking_2', batch_idx, on_step=True, on_epoch=True, sync_dist=True)
                self.dummy_metric(batch_idx)
                self.log('tracking_metric', self.dummy_metric, on_step=True, on_epoch=True)
                value = self.results['training_step.tracking']
                assert value == sum(range(batch_idx + 1))
                value = self.results['training_step.tracking_2']
                assert value == sum(range(batch_idx + 1))
            return super().training_step(batch, batch_idx)

        def on_train_epoch_end(self) -> None:
            if False:
                for i in range(10):
                    print('nop')
            if self.trainer.fit_loop.restarting:
                total = sum(range(self.breaking_batch_idx, batches))
                metrics = self.results.metrics(on_step=False)
                computed_value = self.dummy_metric.compute()
                assert self.results['training_step.tracking'].value == total
                expected = total / (batches - self.breaking_batch_idx)
                assert metrics['callback']['tracking'] == expected
                assert computed_value == 2
                assert self.results['training_step.tracking_2'].value == total
                assert metrics['callback']['tracking_2'] == expected
                assert computed_value == 2
                self.has_validated_sum = True
    model = ExtendedBoringModel()
    trainer_kwargs = {'max_epochs': 1, 'limit_train_batches': batches, 'limit_val_batches': 0, 'accelerator': accelerator, 'devices': devices, 'enable_progress_bar': False, 'enable_model_summary': False, 'default_root_dir': default_root_dir, 'callbacks': OnExceptionCheckpoint(default_root_dir)}
    trainer_kwargs.update(kwargs)
    trainer = Trainer(**trainer_kwargs)
    with suppress(CustomException):
        trainer.fit(model)
    assert not model.has_validated_sum
    tmpdir = trainer.strategy.broadcast(trainer_kwargs['default_root_dir'], 0) if devices >= 2 else trainer_kwargs['default_root_dir']
    ckpt_path = os.path.join(tmpdir, 'on_exception.ckpt')
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, ckpt_path=ckpt_path)
    assert model.has_validated_sum

@pytest.mark.parametrize('kwargs', [{}, pytest.param({'strategy': 'ddp', 'accelerator': 'gpu', 'devices': 1}, marks=RunIf(min_cuda_gpus=1)), pytest.param({'strategy': 'ddp', 'accelerator': 'gpu', 'devices': 2}, marks=RunIf(min_cuda_gpus=2, standalone=True))])
def test_result_collection_reload(tmpdir, kwargs):
    if False:
        for i in range(10):
            print('nop')
    result_collection_reload(default_root_dir=tmpdir, **kwargs)

def test_metric_collections(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'This test ensures the metric attribute is properly found even with complex nested metric structure.'

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.metrics_list = ModuleList([DummyMetric() for _ in range(2)])
            self.metrics_dict = ModuleDict({'a': DummyMetric(), 'b': DummyMetric()})
            self.metrics_collection_dict = MetricCollection({'a': DummyMetric(), 'b': DummyMetric()})
            self.metrics_collection_dict_nested = ModuleDict({'a': ModuleList([ModuleDict({'b': DummyMetric()}), DummyMetric()])})

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            loss = super().training_step(batch, batch_idx)
            self.metrics_list[0](batch_idx)
            self.metrics_list[1](batch_idx)
            self.metrics_dict['a'](batch_idx)
            self.metrics_dict['b'](batch_idx)
            self.metrics_collection_dict['a'](batch_idx)
            self.metrics_collection_dict['b'](batch_idx)
            self.metrics_collection_dict_nested['a'][0]['b'](batch_idx)
            self.metrics_collection_dict_nested['a'][1](batch_idx)
            self.log('a', self.metrics_list[0])
            self.log('b', self.metrics_list[1])
            self.log('c', self.metrics_dict['a'])
            self.log('d', self.metrics_dict['b'])
            self.log('e', self.metrics_collection_dict['a'])
            self.log('f', self.metrics_collection_dict['b'])
            self.log('g', self.metrics_collection_dict_nested['a'][0]['b'])
            self.log('h', self.metrics_collection_dict_nested['a'][1])
            return loss

        def on_train_epoch_end(self) -> None:
            if False:
                i = 10
                return i + 15
            results = self.trainer.fit_loop.epoch_loop._results
            assert results['training_step.a'].meta.metric_attribute == 'metrics_list.0'
            assert results['training_step.b'].meta.metric_attribute == 'metrics_list.1'
            assert results['training_step.c'].meta.metric_attribute == 'metrics_dict.a'
            assert results['training_step.d'].meta.metric_attribute == 'metrics_dict.b'
            assert results['training_step.e'].meta.metric_attribute == 'metrics_collection_dict.a'
            assert results['training_step.f'].meta.metric_attribute == 'metrics_collection_dict.b'
            assert results['training_step.g'].meta.metric_attribute == 'metrics_collection_dict_nested.a.0.b'
            assert results['training_step.h'].meta.metric_attribute == 'metrics_collection_dict_nested.a.1'
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=2, limit_val_batches=0)
    trainer.fit(model)

def test_metric_result_computed_check():
    if False:
        for i in range(10):
            print('nop')
    'Unittest ``_get_cache`` with multielement tensors.'
    metadata = _Metadata('foo', 'bar', on_epoch=True, enable_graph=True)
    metadata.sync = _Sync()
    rm = _ResultMetric(metadata, is_tensor=True)
    computed_value = tensor([1, 2, 3])
    rm._computed = computed_value
    cache = _ResultCollection._get_cache(rm, on_step=False)
    assert cache is computed_value

@pytest.mark.parametrize(('default_type', 'converted_type'), [(torch.half, torch.float), (torch.float, torch.float), (torch.double, torch.double)])
def test_metric_result_respects_dtype(default_type, converted_type):
    if False:
        i = 10
        return i + 15
    from lightning.pytorch.trainer.connectors.logger_connector.result import warning_cache
    warning_cache.clear()
    torch.set_default_dtype(default_type)
    fixed_dtype = torch.long
    metadata = _Metadata('foo', 'bar')
    metadata.sync = _Sync()
    rm = _ResultMetric(metadata, is_tensor=True)
    assert rm.value.dtype == converted_type
    assert rm.cumulated_batch_size.dtype == fixed_dtype
    (value, batch_size) = (tensor(2), 3)
    assert value.dtype == fixed_dtype
    with pytest.warns(UserWarning, match=f"`self.log\\('bar', ...\\)` in your `foo` .* Converting it to {converted_type}"):
        rm.update(value, batch_size)
    rm.update(tensor(4.0), 5)
    total = rm.compute()
    assert total == (2 * 3 + 4 * 5) / (5 + 3)
    assert total.dtype == converted_type
    torch.set_default_dtype(torch.float)

@pytest.mark.parametrize('reduce_fx', ['mean', sum])
def test_metric_result_dtype_promotion(reduce_fx):
    if False:
        print('Hello World!')
    metadata = _Metadata('foo', 'bar', reduce_fx=reduce_fx)
    metadata.sync = _Sync()
    rm = _ResultMetric(metadata, is_tensor=True)
    assert rm.value.dtype == torch.float
    rm.update(tensor(0, dtype=torch.double), 1)
    assert rm.value.dtype == torch.double
    rm.update(tensor(0, dtype=torch.float), 1)
    assert rm.value.dtype == torch.double
    total = rm.compute()
    assert total.dtype == torch.double

@pytest.mark.parametrize('input_dtype', [torch.int8, torch.float16, torch.bfloat16])
def test_metric_result_precision_no_lower_than_float32(input_dtype):
    if False:
        print('Hello World!')
    'Test that the ResultMetric only stores values in float32 or higher precision for numerical stability.'
    metadata = _Metadata('foo', 'bar', reduce_fx='sum')
    metadata.sync = _Sync()
    metric = _ResultMetric(metadata, is_tensor=True)
    assert metric.value.dtype == torch.float
    for i in range(1000):
        metric.update(tensor(1.0, dtype=input_dtype), 1)
        assert metric.value.dtype == torch.float32
    total = metric.compute()
    assert total.item() == 1000.0
    assert total.dtype == torch.float32

@pytest.mark.parametrize(('reduce_fx', 'expected'), [(max, -2), (min, 2)])
def test_result_metric_max_min(reduce_fx, expected):
    if False:
        for i in range(10):
            print('nop')
    metadata = _Metadata('foo', 'bar', reduce_fx=reduce_fx)
    metadata.sync = _Sync()
    rm = _ResultMetric(metadata, is_tensor=True)
    rm.update(tensor(expected), 1)
    assert rm.compute() == expected

def test_compute_not_a_tensor_raises():
    if False:
        for i in range(10):
            print('nop')

    class RandomMetric(Metric):

        def update(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def compute(self):
            if False:
                print('Hello World!')
            return (tensor(1.0), tensor(2.0))

    class MyModel(BoringModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.metric = RandomMetric()

        def on_train_start(self):
            if False:
                for i in range(10):
                    print('nop')
            self.log('foo', self.metric)
    model = MyModel()
    trainer = Trainer(limit_train_batches=1, limit_val_batches=0, max_epochs=1, enable_progress_bar=False, enable_checkpointing=False, logger=False, enable_model_summary=False)
    with pytest.raises(ValueError, match="compute\\(\\)` return of.*foo' must be a tensor"):
        trainer.fit(model)

@pytest.mark.parametrize('distributed_env', [True, False])
@pytest.mark.parametrize('log_val', [tensor(0.5), 'Accuracy'])
def test_logger_sync_dist(distributed_env, log_val):
    if False:
        print('Hello World!')
    if log_val == 'Accuracy':
        log_val = Accuracy(task='binary') if _TM_GE_0_11 else Accuracy()
    pl.trainer.connectors.logger_connector.result.warning_cache.clear()
    meta = _Metadata('foo', 'bar')
    meta.sync = _Sync(_should=False)
    is_tensor = isinstance(log_val, Tensor)
    if not is_tensor:
        log_val.update(tensor([0, 1]), tensor([0, 0], dtype=torch.long))
    result_metric = _ResultMetric(metadata=meta, is_tensor=is_tensor)
    result_metric.update(log_val, 10)
    warning_ctx = pytest.warns if distributed_env and is_tensor else no_warning_call
    patch_ctx = mock.patch('torch.distributed.is_initialized', return_value=distributed_env) if isinstance(log_val, Tensor) else nullcontext()
    with warning_ctx(PossibleUserWarning, match="recommended to use `self.log\\('bar', ..., sync_dist=True\\)`"), patch_ctx:
        value = _ResultCollection._get_cache(result_metric, on_step=False)
    assert value == 0.5