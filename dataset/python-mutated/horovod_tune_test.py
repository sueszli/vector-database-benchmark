import os
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import ray
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
import ray.train.torch
from ray.train.horovod import HorovodTrainer
from ray import train, tune
from ray.tune.schedulers import create_scheduler
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
from ray.tune.utils.release_test_util import ProgressCallback
CIFAR10_STATS = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.201)}

def train_loop_per_worker(config):
    if False:
        while True:
            i = 10
    import horovod.torch as hvd
    hvd.init()
    device = ray.train.torch.get_device()
    net = resnet18().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'])
    epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            model_state = torch.load(checkpoint_dir / 'model.pt', map_location='cpu')
            optimizer_state = torch.load(checkpoint_dir / 'optim.pt', map_location='cpu')
            epoch = torch.load(checkpoint_dir / 'extra_state.pt')['epoch'] + 1
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = hvd.DistributedOptimizer(optimizer)
    np.random.seed(1 + hvd.rank())
    torch.manual_seed(1234)
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    trainset = ray.get(config['data'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = DataLoader(trainset, batch_size=int(config['batch_size']), sampler=train_sampler)
    for epoch in range(epoch, 40):
        running_loss = 0.0
        epoch_steps = 0
        for (i, data) in enumerate(trainloader):
            (inputs, labels) = data
            (inputs, labels) = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / epoch_steps))
            if config['smoke_test']:
                break
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(net.state_dict(), os.path.join(tmpdir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(tmpdir, 'optim.pt'))
            torch.save({'epoch': epoch}, os.path.join(tmpdir, 'extra_state.pt'))
            train.report(dict(loss=running_loss / epoch_steps), checkpoint=Checkpoint.from_directory(tmpdir))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing.')
    args = parser.parse_args()
    if args.smoke_test:
        ray.init()
    else:
        ray.init(address='auto')
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
    dataset = torchvision.datasets.CIFAR10(root='/tmp/data_cifar', train=True, download=True, transform=transform_train)
    horovod_trainer = HorovodTrainer(train_loop_per_worker=train_loop_per_worker, scaling_config=ScalingConfig(use_gpu=False if args.smoke_test else True, num_workers=2), train_loop_config={'batch_size': 64, 'data': ray.put(dataset)})
    pbt = create_scheduler('pbt', perturbation_interval=1, hyperparam_mutations={'train_loop_config': {'lr': tune.uniform(0.001, 0.1)}})
    tuner = Tuner(horovod_trainer, param_space={'train_loop_config': {'lr': 0.1 if args.smoke_test else tune.grid_search([0.1 * i for i in range(1, 5)]), 'smoke_test': args.smoke_test}}, tune_config=TuneConfig(num_samples=2 if args.smoke_test else 1, metric='loss', mode='min', scheduler=pbt), run_config=RunConfig(stop={'training_iteration': 1} if args.smoke_test else None, failure_config=FailureConfig(fail_fast=False), checkpoint_config=CheckpointConfig(num_to_keep=4), callbacks=[ProgressCallback()], storage_path='/mnt/cluster_storage'))
    result_grid = tuner.fit()
    for result in result_grid:
        assert not result.error
    print('Best hyperparameters found were: ', result_grid.get_best_result().config)