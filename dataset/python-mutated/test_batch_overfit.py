import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils

class BatchOverfitCallbackCheck(dl.Callback):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(order=dl.CallbackOrder.external)

    def on_loader_start(self, runner):
        if False:
            for i in range(10):
                print('nop')
        assert len(runner.loaders[runner.loader_key]) == 32

def _prepare_experiment():
    if False:
        print('Hello World!')
    utils.set_global_seed(42)
    (num_samples, num_features) = (int(320.0), int(10.0))
    (X, y) = (torch.rand(num_samples, num_features), torch.rand(num_samples))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    loaders = {'train': loader, 'valid': loader}
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])
    return (loaders, model, criterion, optimizer, scheduler)

def test_batch_overfit():
    if False:
        i = 10
        return i + 15
    (loaders, model, criterion, optimizer, scheduler) = _prepare_experiment()
    runner = dl.SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, loaders=loaders, logdir='./logs/batch_overfit', num_epochs=1, verbose=False, callbacks=[dl.BatchOverfitCallback(train=1, valid=0.1)])
    assert runner.epoch_metrics['train']['loss'] < 1.4
    assert runner.epoch_metrics['valid']['loss'] < 1.3