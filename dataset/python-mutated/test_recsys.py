import os
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import mark
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import IS_CONFIGS_REQUIRED, IS_CPU_REQUIRED, IS_DDP_AMP_REQUIRED, IS_DDP_REQUIRED, IS_DP_AMP_REQUIRED, IS_DP_REQUIRED, IS_GPU_AMP_REQUIRED, IS_GPU_REQUIRED
from tests.misc import run_experiment_from_configs

def train_experiment(engine=None):
    if False:
        print('Hello World!')
    with TemporaryDirectory() as logdir:
        (num_users, num_features, num_items) = (int(10000.0), int(10.0), 10)
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {'train': loader, 'valid': loader}
        model = torch.nn.Linear(num_features, num_items)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        callbacks = [dl.BatchTransformCallback(input_key='logits', output_key='scores', transform=torch.sigmoid, scope='on_batch_end'), dl.CriterionCallback(input_key='logits', target_key='targets', metric_key='loss'), dl.HitrateCallback(input_key='scores', target_key='targets', topk=(1, 3, 5)), dl.MRRCallback(input_key='scores', target_key='targets', topk=(1, 3, 5)), dl.MAPCallback(input_key='scores', target_key='targets', topk=(1, 3, 5)), dl.NDCGCallback(input_key='scores', target_key='targets', topk=(1, 3)), dl.BackwardCallback(metric_key='loss'), dl.OptimizerCallback(metric_key='loss'), dl.SchedulerCallback(), dl.CheckpointCallback(logdir=logdir, loader_key='valid', metric_key='map01', minimize=False)]
        if isinstance(engine, dl.CPUEngine):
            callbacks.append(dl.AUCCallback(input_key='logits', target_key='targets'))
        runner = dl.SupervisedRunner(input_key='features', output_key='logits', target_key='targets', loss_key='loss')
        runner.train(engine=engine, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, loaders=loaders, num_epochs=1, verbose=False, callbacks=callbacks)

def train_experiment_from_configs(*auxiliary_configs: str):
    if False:
        i = 10
        return i + 15
    configs_dir = Path(__file__).parent / 'configs'
    main_config = f'{Path(__file__).stem}.yml'
    d = utils.load_config(str(configs_dir / main_config), ordered=True)['shared']
    X = torch.rand(d['num_users'], d['num_features'])
    y = (torch.rand(d['num_users'], d['num_items']) > 0.5).to(torch.float32)
    torch.save(X, Path('tests') / 'X.pt')
    torch.save(y, Path('tests') / 'y.pt')
    run_experiment_from_configs(configs_dir, main_config, *auxiliary_configs)

@mark.skipif(not IS_CPU_REQUIRED, reason='CUDA device is not available')
def test_run_on_cpu():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.CPUEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not IS_CPU_REQUIRED, reason='CPU device is not available')
def test_config_run_on_cpu():
    if False:
        i = 10
        return i + 15
    train_experiment_from_configs('engine_cpu.yml')

@mark.skipif(not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason='CUDA device is not available')
def test_run_on_torch_cuda0():
    if False:
        print('Hello World!')
    train_experiment(dl.GPUEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason='CUDA device is not available')
def test_config_run_on_torch_cuda0():
    if False:
        while True:
            i = 10
    train_experiment_from_configs('engine_gpu.yml')

@mark.skipif(not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_run_on_amp():
    if False:
        while True:
            i = 10
    train_experiment(dl.GPUEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_config_run_on_amp():
    if False:
        print('Hello World!')
    train_experiment_from_configs('engine_gpu_amp.yml')

@mark.skipif(not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_dp():
    if False:
        return 10
    train_experiment(dl.DataParallelEngine())

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_config_run_on_torch_dp():
    if False:
        print('Hello World!')
    train_experiment_from_configs('engine_dp.yml')

@mark.skipif(not all([IS_DP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_run_on_amp_dp():
    if False:
        return 10
    train_experiment(dl.DataParallelEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_config_run_on_amp_dp():
    if False:
        while True:
            i = 10
    train_experiment_from_configs('engine_dp_amp.yml')

@mark.skipif(not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_ddp():
    if False:
        print('Hello World!')
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
        while True:
            i = 10
    train_experiment(dl.DistributedDataParallelEngine(fp16=True))

@mark.skipif(not IS_CONFIGS_REQUIRED or not all([IS_DDP_AMP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2, SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_config_run_on_amp_ddp():
    if False:
        i = 10
        return i + 15
    train_experiment_from_configs('engine_ddp_amp.yml')

def _train_fn(local_rank, world_size):
    if False:
        return 10
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
        return 10
    world_size: int = torch.cuda.device_count()
    mp.spawn(_train_fn, args=(world_size,), nprocs=world_size, join=True)

def _train_fn_amp(local_rank, world_size):
    if False:
        return 10
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
        print('Hello World!')
    world_size: int = torch.cuda.device_count()
    mp.spawn(_train_fn_amp, args=(world_size,), nprocs=world_size, join=True)