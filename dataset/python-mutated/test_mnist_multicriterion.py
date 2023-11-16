import os
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import mark
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl, metrics
from catalyst.contrib.datasets import MNIST
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import DATA_ROOT, IS_CONFIGS_REQUIRED, IS_CPU_REQUIRED, IS_DDP_AMP_REQUIRED, IS_DDP_REQUIRED, IS_DP_AMP_REQUIRED, IS_DP_REQUIRED, IS_GPU_AMP_REQUIRED, IS_GPU_REQUIRED
from tests.misc import run_experiment_from_configs

class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        if False:
            return 10
        return self.model(batch[0].to(self.device))

    def on_loader_start(self, runner):
        if False:
            print('Hello World!')
        super().on_loader_start(runner)
        self.meters = {key: metrics.AdditiveMetric(compute_on_call=False) for key in ['loss', 'accuracy01', 'accuracy03']}

    def handle_batch(self, batch):
        if False:
            i = 10
            return i + 15
        (x, y) = batch
        logits = self.model(x)
        loss_multiclass = self.criterion['multiclass'](logits, y)
        loss_multilabel = self.criterion['multilabel'](logits, F.one_hot(y, 10).to(torch.float32))
        loss = loss_multiclass + loss_multilabel
        (accuracy01, accuracy03) = metrics.accuracy(logits, y, topk=(1, 3))
        self.batch_metrics.update({'loss': loss, 'accuracy01': accuracy01, 'accuracy03': accuracy03})
        for key in ['loss', 'accuracy01', 'accuracy03']:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        if False:
            return 10
        for key in ['loss', 'accuracy01', 'accuracy03']:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

def train_experiment(engine=None):
    if False:
        i = 10
        return i + 15
    with TemporaryDirectory() as logdir:
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        criterion = {'multiclass': nn.CrossEntropyLoss(), 'multilabel': nn.BCEWithLogitsLoss()}
        loaders = {'train': DataLoader(MNIST(DATA_ROOT, train=True), batch_size=32), 'valid': DataLoader(MNIST(DATA_ROOT, train=False), batch_size=32)}
        runner = CustomRunner()
        runner.train(engine=engine, model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=1, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True)

def train_experiment_from_configs(*auxiliary_configs: str):
    if False:
        i = 10
        return i + 15
    run_experiment_from_configs(Path(__file__).parent / 'configs', f'{Path(__file__).stem}.yml', *auxiliary_configs)

@mark.skipif(not IS_CPU_REQUIRED, reason='CUDA device is not available')
def test_run_on_cpu():
    if False:
        for i in range(10):
            print('nop')
    train_experiment(dl.CPUEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not IS_CPU_REQUIRED, reason='CPU device is not available')
def test_config_run_on_cpu():
    if False:
        while True:
            i = 10
    train_experiment_from_configs('engine_cpu.yml')

@mark.skipif(not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason='CUDA device is not available')
def test_run_on_torch_cuda0():
    if False:
        for i in range(10):
            print('nop')
    train_experiment(dl.GPUEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason='CUDA device is not available')
def test_config_run_on_torch_cuda0():
    if False:
        for i in range(10):
            print('nop')
    train_experiment_from_configs('engine_gpu.yml')

@mark.skipif(not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_run_on_amp():
    if False:
        return 10
    train_experiment(dl.GPUEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_config_run_on_amp():
    if False:
        return 10
    train_experiment_from_configs('engine_gpu_amp.yml')

@mark.skipif(not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_dp():
    if False:
        return 10
    train_experiment(dl.DataParallelEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_config_run_on_torch_dp():
    if False:
        while True:
            i = 10
    train_experiment_from_configs('engine_dp.yml')

@mark.skipif(not all([IS_DP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_run_on_amp_dp():
    if False:
        for i in range(10):
            print('nop')
    train_experiment(dl.DataParallelEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_config_run_on_amp_dp():
    if False:
        for i in range(10):
            print('nop')
    train_experiment_from_configs('engine_dp_amp.yml')