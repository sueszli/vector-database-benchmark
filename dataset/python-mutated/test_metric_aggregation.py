import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils
from catalyst.contrib.losses import FocalLossBinary

def prepare_experiment():
    if False:
        while True:
            i = 10
    utils.set_global_seed(42)
    (num_samples, num_features, num_classes) = (int(10000.0), int(10.0), 4)
    X = torch.rand(num_samples, num_features)
    y = (torch.rand(num_samples) * num_classes).to(torch.int64)
    y = torch.nn.functional.one_hot(y, num_classes).double()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, num_classes)
    criterion = {'bce': torch.nn.BCEWithLogitsLoss(), 'focal': FocalLossBinary()}
    optimizer = torch.optim.Adam(model.parameters())
    return (loaders, model, criterion, optimizer)

def test_aggregation_1():
    if False:
        for i in range(10):
            print('nop')
    '\n    Aggregation as weighted_sum\n    '
    (loaders, model, criterion, optimizer) = prepare_experiment()
    runner = dl.SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir='./logs/aggregation_1/', num_epochs=3, callbacks=[dl.CriterionCallback(input_key='logits', target_key='targets', metric_key='loss_bce', criterion_key='bce'), dl.CriterionCallback(input_key='logits', target_key='targets', metric_key='loss_focal', criterion_key='focal'), dl.MetricAggregationCallback(metric_key='loss', metrics={'loss_focal': 0.6, 'loss_bce': 0.4}, mode='weighted_sum')])
    for loader in ['train', 'valid']:
        metrics = runner.epoch_metrics[loader]
        loss_1 = metrics['loss_bce'] * 0.4 + metrics['loss_focal'] * 0.6
        loss_2 = metrics['loss']
        assert np.abs(loss_1 - loss_2) < 1e-05

def test_aggregation_2():
    if False:
        for i in range(10):
            print('nop')
    '\n    Aggregation with custom function\n    '
    (loaders, model, criterion, optimizer) = prepare_experiment()
    runner = dl.SupervisedRunner()

    def aggregation_function(metrics, runner):
        if False:
            print('Hello World!')
        epoch = runner.epoch_step
        loss = (3 / 2 - epoch / 2) * metrics['loss_focal'] + (1 / 2 * epoch - 1 / 2) * metrics['loss_bce']
        return loss
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, logdir='./logs/aggregation_2/', num_epochs=3, callbacks=[dl.CriterionCallback(input_key='logits', target_key='targets', metric_key='loss_bce', criterion_key='bce'), dl.CriterionCallback(input_key='logits', target_key='targets', metric_key='loss_focal', criterion_key='focal'), dl.MetricAggregationCallback(metric_key='loss', mode=aggregation_function)])
    for loader in ['train', 'valid']:
        metrics = runner.epoch_metrics[loader]
        loss_1 = metrics['loss_bce']
        loss_2 = metrics['loss']
        assert np.abs(loss_1 - loss_2) < 1e-05