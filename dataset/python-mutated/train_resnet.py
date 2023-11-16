from typing import Any, Optional
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from catalyst import dl
from catalyst.contrib.data import Compose, ImageToTensor, NormalizeImage
from catalyst.contrib.datasets import CIFAR10
from catalyst.contrib.layers import ResidualBlock
from src import E2E, parse_params

def conv_block(in_channels, out_channels, pool=False):
    if False:
        print('Hello World!')
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def resnet9(in_channels: int, num_classes: int, size: int=16):
    if False:
        while True:
            i = 10
    (sz, sz2, sz4, sz8) = (size, size * 2, size * 4, size * 8)
    return nn.Sequential(conv_block(in_channels, sz), conv_block(sz, sz2, pool=True), ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))), conv_block(sz2, sz4, pool=True), conv_block(sz4, sz8, pool=True), ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))), nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(sz8, num_classes)))

class CustomRunner(dl.IRunner):

    def __init__(self, logdir: str, engine: str, **engine_params: Any):
        if False:
            while True:
                i = 10
        super().__init__()
        self._logdir = logdir
        self._engine = engine
        self._engine_params = engine_params

    def get_engine(self):
        if False:
            print('Hello World!')
        return E2E[self._engine](**self._engine_params)

    def get_loggers(self):
        if False:
            i = 10
            return i + 15
        return {'console': dl.ConsoleLogger(), 'csv': dl.CSVLogger(logdir=self._logdir), 'tensorboard': dl.TensorboardLogger(logdir=self._logdir)}

    @property
    def num_epochs(self) -> int:
        if False:
            return 10
        return 10

    def get_loaders(self):
        if False:
            return 10
        transform = Compose([ImageToTensor(), NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        valid_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(train_data, num_replicas=self.engine.num_processes, rank=self.engine.process_index, shuffle=True)
            valid_sampler = DistributedSampler(valid_data, num_replicas=self.engine.num_processes, rank=self.engine.process_index, shuffle=False)
        else:
            train_sampler = valid_sampler = None
        return {'train': DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=4), 'valid': DataLoader(valid_data, batch_size=32, sampler=valid_sampler, num_workers=4)}

    def get_model(self):
        if False:
            i = 10
            return i + 15
        model = self.model if self.model is not None else resnet9(in_channels=3, num_classes=10)
        return model

    def get_criterion(self):
        if False:
            return 10
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model):
        if False:
            i = 10
            return i + 15
        return optim.Adam(model.parameters(), lr=0.001)

    def get_scheduler(self, optimizer):
        if False:
            print('Hello World!')
        return optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

    def get_callbacks(self):
        if False:
            return 10
        return {'criterion': dl.CriterionCallback(metric_key='loss', input_key='logits', target_key='targets'), 'backward': dl.BackwardCallback(metric_key='loss'), 'optimizer': dl.OptimizerCallback(metric_key='loss'), 'scheduler': dl.SchedulerCallback(loader_key='valid', metric_key='loss'), 'accuracy': dl.AccuracyCallback(input_key='logits', target_key='targets', topk=(1, 3, 5)), 'checkpoint': dl.CheckpointCallback(self._logdir, loader_key='valid', metric_key='accuracy01', minimize=False, topk=1), 'tqdm': dl.TqdmCallback()}

    def handle_batch(self, batch):
        if False:
            print('Hello World!')
        (x, y) = batch
        logits = self.model(x)
        self.batch = {'features': x, 'targets': y, 'logits': logits}
if __name__ == '__main__':
    (kwargs, _) = parse_params('resnet')
    runner = CustomRunner(**kwargs)
    runner.run()