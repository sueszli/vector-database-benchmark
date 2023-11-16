import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, ToTensor
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy
in_colab = 'COLAB_TPU_ADDR' in os.environ
with_torchrun = 'WORLD_SIZE' in os.environ
train_transform = Compose([Pad(4), RandomCrop(32, fill=128), RandomHorizontalFlip(), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225))])
test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225))])

def get_train_test_datasets(path):
    if False:
        for i in range(10):
            print('nop')
    if idist.get_rank() > 0:
        idist.barrier()
    train_ds = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)
    if idist.get_rank() == 0:
        idist.barrier()
    return (train_ds, test_ds)

def get_model(name):
    if False:
        while True:
            i = 10
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f'Unknown model name {name}')
    return fn(num_classes=10)

def get_dataflow(config):
    if False:
        while True:
            i = 10
    (train_dataset, test_dataset) = get_train_test_datasets(config.get('data_path', '.'))
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config.get('batch_size', 512), num_workers=config.get('num_workers', 8), shuffle=True, drop_last=True)
    config['num_iters_per_epoch'] = len(train_loader)
    test_loader = idist.auto_dataloader(test_dataset, batch_size=2 * config.get('batch_size', 512), num_workers=config.get('num_workers', 8), shuffle=False)
    return (train_loader, test_loader)

def initialize(config):
    if False:
        while True:
            i = 10
    model = get_model(config['model'])
    model = idist.auto_model(model)
    optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate', 0.1), momentum=config.get('momentum', 0.9), weight_decay=config.get('weight_decay', 1e-05), nesterov=True)
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.CrossEntropyLoss().to(idist.device())
    le = config['num_iters_per_epoch']
    lr_scheduler = StepLR(optimizer, step_size=le, gamma=0.9)
    return (model, optimizer, criterion, lr_scheduler)

def create_trainer(model, optimizer, criterion, lr_scheduler, config):
    if False:
        i = 10
        return i + 15

    def train_step(engine, batch):
        if False:
            while True:
                i = 10
        (x, y) = (batch[0].to(idist.device()), batch[1].to(idist.device()))
        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        return loss.item()
    trainer = Engine(train_step)
    if idist.get_rank() == 0:

        @trainer.on(Events.ITERATION_COMPLETED(every=200))
        def save_checkpoint():
            if False:
                print('Hello World!')
            fp = Path(config.get('output_path', 'output')) / 'checkpoint.pt'
            torch.save(model.state_dict(), fp)
        ProgressBar().attach(trainer, output_transform=lambda x: {'batch loss': x})
    return trainer

def training(local_rank, config):
    if False:
        return 10
    (train_loader, val_loader) = get_dataflow(config)
    (model, optimizer, criterion, lr_scheduler) = initialize(config)
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, config)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()}, device=idist.device())

    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def evaluate_model():
        if False:
            print('Hello World!')
        state = evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print(state.metrics)
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(config.get('output_path', 'output'), trainer, optimizer, evaluators={'validation': evaluator})
    trainer.run(train_loader, max_epochs=config.get('max_epochs', 3))
    if idist.get_rank() == 0:
        tb_logger.close()
if __name__ == '__main__' and (not (in_colab or with_torchrun)):
    backend = None
    nproc_per_node = None
    config = {'model': 'resnet18', 'dataset': 'cifar10'}
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)
if __name__ == '__main__' and with_torchrun:
    backend = 'nccl'
    nproc_per_node = None
    config = {'model': 'resnet18', 'dataset': 'cifar10'}
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)
if in_colab:
    backend = 'xla-tpu'
    nproc_per_node = 8
    config = {'model': 'resnet18', 'dataset': 'cifar10'}
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)