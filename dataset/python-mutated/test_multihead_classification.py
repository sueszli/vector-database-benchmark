import os
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import mark
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import IS_CONFIGS_REQUIRED, IS_CPU_REQUIRED, IS_DDP_AMP_REQUIRED, IS_DDP_REQUIRED, IS_DP_AMP_REQUIRED, IS_DP_REQUIRED, IS_GPU_AMP_REQUIRED, IS_GPU_REQUIRED
from tests.misc import run_experiment_from_configs

class CustomRunner(dl.Runner):

    def handle_batch(self, batch):
        if False:
            return 10
        (x, y1, y2) = batch
        (y1_hat, y2_hat) = self.model(x)
        self.batch = {'features': x, 'logits1': y1_hat, 'logits2': y2_hat, 'targets1': y1, 'targets2': y2}

class CustomModule(nn.Module):

    def __init__(self, in_features: int, out_features1: int, out_features2: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.shared = nn.Linear(in_features, 128)
        self.head1 = nn.Linear(128, out_features1)
        self.head2 = nn.Linear(128, out_features2)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.shared(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        return (y1, y2)

def train_experiment(engine=None):
    if False:
        print('Hello World!')
    with TemporaryDirectory() as logdir:
        (num_samples, num_features, num_classes1, num_classes2) = (int(10000.0), int(10.0), 4, 10)
        X = torch.rand(num_samples, num_features)
        y1 = (torch.rand(num_samples) * num_classes1).to(torch.int64)
        y2 = (torch.rand(num_samples) * num_classes2).to(torch.int64)
        dataset = TensorDataset(X, y1, y2)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {'train': loader, 'valid': loader}
        model = CustomModule(num_features, num_classes1, num_classes2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2])
        callbacks = [dl.CriterionCallback(metric_key='loss1', input_key='logits1', target_key='targets1'), dl.CriterionCallback(metric_key='loss2', input_key='logits2', target_key='targets2'), dl.MetricAggregationCallback(metric_key='loss', metrics=['loss1', 'loss2'], mode='mean'), dl.BackwardCallback(metric_key='loss'), dl.OptimizerCallback(metric_key='loss'), dl.SchedulerCallback(), dl.AccuracyCallback(input_key='logits1', target_key='targets1', num_classes=num_classes1, prefix='one_'), dl.AccuracyCallback(input_key='logits2', target_key='targets2', num_classes=num_classes2, prefix='two_'), dl.CheckpointCallback('./logs/one', loader_key='valid', metric_key='one_accuracy01', minimize=False, topk=1), dl.CheckpointCallback('./logs/two', loader_key='valid', metric_key='two_accuracy03', minimize=False, topk=3)]
        if SETTINGS.ml_required:
            callbacks.append(dl.ConfusionMatrixCallback(input_key='logits1', target_key='targets1', num_classes=num_classes1, prefix='one_cm'))
            callbacks.append(dl.ConfusionMatrixCallback(input_key='logits2', target_key='targets2', num_classes=num_classes2, prefix='two_cm'))
        runner = CustomRunner()
        runner.train(engine=engine, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, loaders=loaders, num_epochs=1, verbose=False, callbacks=callbacks, loggers={'console': dl.ConsoleLogger(), 'tb': dl.TensorboardLogger('./logs/tb')})

def train_experiment_from_configs(*auxiliary_configs: str):
    if False:
        return 10
    configs_dir = Path(__file__).parent / 'configs'
    main_config = f'{Path(__file__).stem}.yml'
    d = utils.load_config(str(configs_dir / main_config), ordered=True)['shared']
    X = torch.rand(d['num_samples'], d['num_features'])
    y1 = (torch.rand(d['num_samples']) * d['num_classes1']).to(torch.int64)
    y2 = (torch.rand(d['num_samples']) * d['num_classes2']).to(torch.int64)
    torch.save(X, Path('tests') / 'X.pt')
    torch.save(y1, Path('tests') / 'y1.pt')
    torch.save(y2, Path('tests') / 'y2.pt')
    run_experiment_from_configs(configs_dir, main_config, *auxiliary_configs)

@mark.skipif(not IS_CPU_REQUIRED, reason='CUDA device is not available')
def test_run_on_cpu():
    if False:
        print('Hello World!')
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
        print('Hello World!')
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
        for i in range(10):
            print('nop')
    train_experiment(dl.GPUEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_config_run_on_amp():
    if False:
        print('Hello World!')
    train_experiment_from_configs('engine_gpu_amp.yml')

@mark.skipif(not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_dp():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.DataParallelEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_config_run_on_torch_dp():
    if False:
        print('Hello World!')
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

@mark.skipif(not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_ddp():
    if False:
        return 10
    train_experiment(dl.DistributedDataParallelEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_config_run_on_torch_ddp():
    if False:
        i = 10
        return i + 15
    train_experiment_from_configs('engine_ddp.yml')

@mark.skipif(not all([IS_DDP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_run_on_amp_ddp():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.DistributedDataParallelEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DDP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_config_run_on_amp_ddp():
    if False:
        print('Hello World!')
    train_experiment_from_configs('engine_ddp_amp.yml')

def _train_fn(local_rank, world_size):
    if False:
        print('Hello World!')
    process_group_kwargs = {'backend': 'nccl', 'world_size': world_size}
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    dist.init_process_group(**process_group_kwargs)
    train_experiment(dl.Engine())
    dist.destroy_process_group()

@mark.skipif(not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_ddp_spawn():
    if False:
        while True:
            i = 10
    world_size: int = torch.cuda.device_count()
    mp.spawn(_train_fn, args=(world_size,), nprocs=world_size, join=True)

def _train_fn_amp(local_rank, world_size):
    if False:
        for i in range(10):
            print('nop')
    process_group_kwargs = {'backend': 'nccl', 'world_size': world_size}
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    dist.init_process_group(**process_group_kwargs)
    train_experiment(dl.Engine(fp16=True))
    dist.destroy_process_group()

@mark.skipif(not all([IS_DDP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_run_on_torch_ddp_amp_spawn():
    if False:
        for i in range(10):
            print('nop')
    world_size: int = torch.cuda.device_count()
    mp.spawn(_train_fn_amp, args=(world_size,), nprocs=world_size, join=True)