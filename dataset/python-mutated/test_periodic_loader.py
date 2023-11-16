import copy
from io import StringIO
import os
import re
import shutil
import sys
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import Callback, CallbackOrder, CheckpointCallback, CheckRunCallback, CriterionCallback, PeriodicLoaderCallback, SupervisedRunner

def test_validation_with_period_3():
    if False:
        for i in range(10):
            print('nop')
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, valid=3), CheckRunCallback(num_epoch_steps=10)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/model.0009.pth')
    assert os.path.isfile(checkpoint + '/model.best.pth')
    assert os.path.isfile(checkpoint + '/model.last.pth')
    shutil.rmtree(logdir, ignore_errors=True)

@pytest.mark.skip(reason='disabled support period = 0 for validation loaders')
def test_validation_with_period_0():
    if False:
        return 10
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=5, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, valid=0), CheckRunCallback(num_epoch_steps=5)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/train.5.pth')
    assert os.path.isfile(checkpoint + '/train.5_full.pth')
    assert os.path.isfile(checkpoint + '/best.pth')
    assert os.path.isfile(checkpoint + '/best_full.pth')
    assert os.path.isfile(checkpoint + '/last.pth')
    assert os.path.isfile(checkpoint + '/last_full.pth')
    shutil.rmtree(logdir, ignore_errors=True)

def test_multiple_loaders():
    if False:
        while True:
            i = 10
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train_additional=2, valid=3, valid_additional=0), CheckRunCallback(num_epoch_steps=10)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/model.0009.pth')
    assert os.path.isfile(checkpoint + '/model.best.pth')
    assert os.path.isfile(checkpoint + '/model.last.pth')
    shutil.rmtree(logdir, ignore_errors=True)

def test_no_loaders_epoch():
    if False:
        i = 10
        return i + 15
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    with pytest.raises(ValueError):
        runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train=2, train_additional=2, valid=3, valid_additional=0)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    shutil.rmtree(logdir, ignore_errors=True)

def test_wrong_period_type():
    if False:
        return 10
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    with pytest.raises(TypeError):
        runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train_additional=[], train_not_exists=2, valid=3, valid_additional=0, valid_not_exist=1)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    shutil.rmtree(logdir, ignore_errors=True)

def test_negative_period_exception():
    if False:
        i = 10
        return i + 15
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    with pytest.raises(ValueError):
        runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train_additional=1, train_not_exists=-10, valid=3, valid_additional=-1, valid_not_exist=1)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    shutil.rmtree(logdir, ignore_errors=True)

def test_zero_period_validation_exception():
    if False:
        for i in range(10):
            print('nop')
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    with pytest.raises(ValueError):
        runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train_additional=1, train_not_exists=3, valid=0, valid_additional=2, valid_not_exist=1)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    shutil.rmtree(logdir, ignore_errors=True)

def test_ignoring_unknown_loaders():
    if False:
        return 10
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'train_additional': loader, 'valid': loader, 'valid_additional': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=10, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, train_additional=2, train_not_exists=2, valid=3, valid_additional=0, valid_not_exist=1), CheckRunCallback(num_epoch_steps=10)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/model.0009.pth')
    assert os.path.isfile(checkpoint + '/model.best.pth')
    assert os.path.isfile(checkpoint + '/model.last.pth')
    shutil.rmtree(logdir, ignore_errors=True)

def test_loading_best_state_at_end():
    if False:
        print('Hello World!')
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir + '/checkpoints'
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=5, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, valid=3), CheckRunCallback(num_epoch_steps=5)], load_best_on_end=True)
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/model.0003.pth')
    assert os.path.isfile(checkpoint + '/model.best.pth')
    assert os.path.isfile(checkpoint + '/model.last.pth')
    shutil.rmtree(logdir, ignore_errors=True)

def test_multiple_best_checkpoints():
    if False:
        return 10
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()
    logdir = './logs/periodic_loader'
    checkpoint = logdir
    logfile = checkpoint + '/model.storage.json'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()
    n_epochs = 12
    period = 2
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir=logdir, num_epochs=n_epochs, verbose=False, valid_loader='valid', valid_metric='loss', minimize_valid_metric=True, callbacks=[PeriodicLoaderCallback(valid_loader_key='valid', valid_metric_key='loss', minimize=True, valid=period), CheckRunCallback(num_epoch_steps=n_epochs), CheckpointCallback(logdir=logdir, loader_key='valid', metric_key='loss', minimize=True, topk=3)])
    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()
    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + '/model.0008.pth')
    assert os.path.isfile(checkpoint + '/model.0010.pth')
    assert os.path.isfile(checkpoint + '/model.0012.pth')
    assert os.path.isfile(checkpoint + '/model.best.pth')
    assert os.path.isfile(checkpoint + '/model.last.pth')
    shutil.rmtree(logdir, ignore_errors=True)