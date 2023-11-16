import csv
import os
import re
from typing import Dict, Optional
from unittest import mock
from unittest.mock import Mock
import pytest
import torch
from lightning.fabric.accelerators.xla import _using_pjrt
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators.cpu import _CPU_PERCENT, _CPU_SWAP_PERCENT, _CPU_VM_PERCENT, get_cpu_stats
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.device_stats_monitor import _prefix_metric_keys
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from tests_pytorch.helpers.runif import RunIf

@RunIf(min_cuda_gpus=1)
def test_device_stats_gpu_from_torch(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test GPU stats are logged using a logger.'
    model = BoringModel()
    device_stats = DeviceStatsMonitor()

    class DebugLogger(CSVLogger):

        @rank_zero_only
        def log_metrics(self, metrics: Dict[str, float], step: Optional[int]=None) -> None:
            if False:
                return 10
            fields = ['allocated_bytes.all.freed', 'inactive_split.all.peak', 'reserved_bytes.large_pool.peak']
            for f in fields:
                assert any((f in h for h in metrics))
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=7, log_every_n_steps=1, accelerator='gpu', devices=1, callbacks=[device_stats], logger=DebugLogger(tmpdir), enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model)

@RunIf(psutil=True)
@pytest.mark.parametrize('cpu_stats', [None, True, False])
@mock.patch('lightning.pytorch.accelerators.cpu.get_cpu_stats', side_effect=get_cpu_stats)
def test_device_stats_cpu(cpu_stats_mock, tmpdir, cpu_stats):
    if False:
        for i in range(10):
            print('nop')
    'Test CPU stats are logged when no accelerator is used.'
    model = BoringModel()
    CPU_METRIC_KEYS = (_CPU_VM_PERCENT, _CPU_SWAP_PERCENT, _CPU_PERCENT)

    class DebugLogger(CSVLogger):

        def log_metrics(self, metrics: Dict[str, float], step: Optional[int]=None) -> None:
            if False:
                for i in range(10):
                    print('nop')
            enabled = cpu_stats is not False
            for f in CPU_METRIC_KEYS:
                has_cpu_metrics = any((f in h for h in metrics))
                assert has_cpu_metrics if enabled else not has_cpu_metrics
    device_stats = DeviceStatsMonitor(cpu_stats=cpu_stats)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, limit_val_batches=0, log_every_n_steps=1, callbacks=device_stats, logger=DebugLogger(tmpdir), enable_checkpointing=False, enable_progress_bar=False, accelerator='cpu')
    trainer.fit(model)
    expected = 4 if cpu_stats is not False else 0
    assert cpu_stats_mock.call_count == expected

class AssertTpuMetricsLogger(CSVLogger):

    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        if False:
            return 10
        fields = ['avg. free memory (MB)', 'avg. peak memory (MB)']
        for f in fields:
            assert any((f in h for h in metrics))

@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_device_stats_monitor_tpu(tmpdir):
    if False:
        return 10
    'Test TPU stats are logged using a logger.'
    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=5, accelerator='tpu', devices='auto', log_every_n_steps=1, callbacks=[device_stats], logger=AssertTpuMetricsLogger(tmpdir), enable_checkpointing=False, enable_progress_bar=False)
    try:
        trainer.fit(model)
    except RuntimeError as e:
        if _using_pjrt() and 'GetMemoryInfo not implemented' in str(e):
            pytest.xfail('`xm.get_memory_info` is not implemented with PJRT')
        raise e

def test_device_stats_monitor_no_logger(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test DeviceStatsMonitor with no logger in Trainer.'
    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[device_stats], max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    with pytest.raises(MisconfigurationException, match='Cannot use `DeviceStatsMonitor` callback.'):
        trainer.fit(model)

def test_prefix_metric_keys():
    if False:
        i = 10
        return i + 15
    'Test that metric key names are converted correctly.'
    metrics = {'1': 1.0, '2': 2.0, '3': 3.0}
    prefix = 'foo'
    separator = '.'
    converted_metrics = _prefix_metric_keys(metrics, prefix, separator)
    assert converted_metrics == {'foo.1': 1.0, 'foo.2': 2.0, 'foo.3': 3.0}

def test_device_stats_monitor_warning_when_psutil_not_available(monkeypatch, tmp_path):
    if False:
        return 10
    'Test that warning is raised when psutil is not available.'
    import lightning.pytorch.callbacks.device_stats_monitor as imports
    monkeypatch.setattr(imports, '_PSUTIL_AVAILABLE', False)
    monitor = DeviceStatsMonitor()
    trainer = Trainer(logger=CSVLogger(tmp_path))
    assert trainer.strategy.root_device == torch.device('cpu')
    with pytest.raises(ModuleNotFoundError, match='psutil` is not installed'):
        monitor.setup(trainer, Mock(), 'fit')

def test_device_stats_monitor_logs_for_different_stages(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that metrics are logged for all stages that is training, testing and validation.'
    model = BoringModel()
    device_stats = DeviceStatsMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=4, limit_val_batches=4, limit_test_batches=1, log_every_n_steps=1, accelerator='cpu', devices=1, callbacks=[device_stats], logger=CSVLogger(tmpdir), enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model)
    with open(f'{tmpdir}/lightning_logs/version_0/metrics.csv') as csvfile:
        content = csv.reader(csvfile, delimiter=',')
        it = iter(content).__next__()
    train_stage_results = [re.match('.+on_train_batch', i) for i in it]
    train = any(train_stage_results)
    assert train, 'training stage logs not found'
    validation_stage_results = [re.match('.+on_validation_batch', i) for i in it]
    valid = any(validation_stage_results)
    assert valid, 'validation stage logs not found'
    trainer.test(model)
    with open(f'{tmpdir}/lightning_logs/version_0/metrics.csv') as csvfile:
        content = csv.reader(csvfile, delimiter=',')
        it = iter(content).__next__()
    test_stage_results = [re.match('.+on_test_batch', i) for i in it]
    test = any(test_stage_results)
    assert test, 'testing stage logs not found'