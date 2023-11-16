import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
from catalyst.dl import Callback, CallbackOrder, SupervisedRunner

class LRCheckerCallback(Callback):

    def __init__(self, init_lr_value: float, final_lr_value: float):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(CallbackOrder.Internal)
        self.init_lr = init_lr_value
        self.final_lr = final_lr_value

    def on_batch_start(self, runner):
        if False:
            i = 10
            return i + 15
        step = getattr(runner, 'batch_step')
        if step == 1:
            assert self.init_lr == runner.scheduler.get_lr()[0]

    def on_experiment_end(self, runner):
        if False:
            i = 10
            return i + 15
        assert self.final_lr == runner.scheduler.get_lr()[0]

def test_onecyle():
    if False:
        for i in range(10):
            print('nop')
    logdir = './logs/core_runner'
    (num_samples, num_features) = (int(10000.0), int(10.0))
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {'train': loader, 'valid': loader}
    num_steps = 6
    epochs = 8
    min_lr = 0.0001
    max_lr = 0.002
    init_lr = 0.001
    warmup_fraction = 0.5
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = OneCycleLRWithWarmup(optimizer, num_steps=num_steps, lr_range=(max_lr, min_lr), init_lr=init_lr, warmup_fraction=warmup_fraction)
    runner = SupervisedRunner()
    callbacks = [LRCheckerCallback(init_lr, min_lr)]
    runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, loaders=loaders, logdir=logdir, num_epochs=epochs, verbose=False, callbacks=callbacks)