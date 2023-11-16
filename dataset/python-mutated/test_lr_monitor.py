import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.finetuning import BackboneFinetuning
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import optim
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel

def test_lr_monitor_single_lr(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that learning rates are extracted and logged for single lr scheduler.'
    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert lr_monitor.lrs, 'No learning rates logged'
    assert all((v is None for v in lr_monitor.last_momentum_values.values())), 'Momentum should not be logged by default'
    assert len(lr_monitor.lrs) == len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ['lr-SGD']

@pytest.mark.parametrize('opt', ['SGD', 'Adam'])
def test_lr_monitor_single_lr_with_momentum(tmpdir, opt: str):
    if False:
        print('Hello World!')
    'Test that learning rates, momentum and weight decay are extracted and logged for single lr scheduler.'

    class LogMomentumModel(BoringModel):

        def __init__(self, opt):
            if False:
                return 10
            super().__init__()
            self.opt = opt

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.opt == 'SGD':
                opt_kwargs = {'momentum': 0.9}
            elif self.opt == 'Adam':
                opt_kwargs = {'betas': (0.9, 0.999)}
            optimizer = getattr(optim, self.opt)(self.parameters(), lr=0.01, **opt_kwargs)
            lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=10000)
            return ([optimizer], [lr_scheduler])
    model = LogMomentumModel(opt=opt)
    lr_monitor = LearningRateMonitor(log_momentum=True, log_weight_decay=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=2, limit_train_batches=5, log_every_n_steps=1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert all((v is not None for v in lr_monitor.last_momentum_values.values())), 'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_scheduler_configs)
    assert all((k == f'lr-{opt}-momentum' for k in lr_monitor.last_momentum_values))
    assert all((v is not None for v in lr_monitor.last_weight_decay_values.values())), 'Expected weight decay to be logged'
    assert len(lr_monitor.last_weight_decay_values) == len(trainer.lr_scheduler_configs)
    assert all((k == f'lr-{opt}-weight_decay' for k in lr_monitor.last_weight_decay_values))

def test_log_momentum_no_momentum_optimizer(tmpdir):
    if False:
        i = 10
        return i + 15
    "Test that if optimizer doesn't have momentum then a warning is raised with log_momentum=True."

    class LogMomentumModel(BoringModel):

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            optimizer = optim.ASGD(self.parameters(), lr=0.01)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return ([optimizer], [lr_scheduler])
    model = LogMomentumModel()
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=2, limit_train_batches=5, log_every_n_steps=1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    with pytest.warns(RuntimeWarning, match='optimizers do not have momentum.'):
        trainer.fit(model)
    assert all((v == 0 for v in lr_monitor.last_momentum_values.values())), 'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.lr_scheduler_configs)
    assert all((k == 'lr-ASGD-momentum' for k in lr_monitor.last_momentum_values))

def test_lr_monitor_no_lr_scheduler_single_lr(tmpdir):
    if False:
        print('Hello World!')
    'Test that learning rates are extracted and logged for no lr scheduler.'

    class CustomBoringModel(BoringModel):

        def configure_optimizers(self):
            if False:
                i = 10
                return i + 15
            return optim.SGD(self.parameters(), lr=0.1)
    model = CustomBoringModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == len(trainer.optimizers)
    assert list(lr_monitor.lrs) == ['lr-SGD']

@pytest.mark.parametrize('opt', ['SGD', 'Adam'])
def test_lr_monitor_no_lr_scheduler_single_lr_with_momentum(tmpdir, opt: str):
    if False:
        i = 10
        return i + 15
    'Test that learning rates and momentum are extracted and logged for no lr scheduler.'

    class LogMomentumModel(BoringModel):

        def __init__(self, opt):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.opt = opt

        def configure_optimizers(self):
            if False:
                i = 10
                return i + 15
            if self.opt == 'SGD':
                opt_kwargs = {'momentum': 0.9}
            elif self.opt == 'Adam':
                opt_kwargs = {'betas': (0.9, 0.999)}
            optimizer = getattr(optim, self.opt)(self.parameters(), lr=0.01, **opt_kwargs)
            return [optimizer]
    model = LogMomentumModel(opt=opt)
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=2, limit_train_batches=5, log_every_n_steps=1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert all((v is not None for v in lr_monitor.last_momentum_values.values())), 'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.optimizers)
    assert all((k == f'lr-{opt}-momentum' for k in lr_monitor.last_momentum_values))

def test_log_momentum_no_momentum_optimizer_no_lr_scheduler(tmpdir):
    if False:
        print('Hello World!')
    "Test that if optimizer doesn't have momentum then a warning is raised with log_momentum=True."

    class LogMomentumModel(BoringModel):

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            optimizer = optim.ASGD(self.parameters(), lr=0.01)
            return [optimizer]
    model = LogMomentumModel()
    lr_monitor = LearningRateMonitor(log_momentum=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=2, limit_train_batches=5, log_every_n_steps=1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    with pytest.warns(RuntimeWarning, match='optimizers do not have momentum.'):
        trainer.fit(model)
    assert all((v == 0 for v in lr_monitor.last_momentum_values.values())), 'Expected momentum to be logged'
    assert len(lr_monitor.last_momentum_values) == len(trainer.optimizers)
    assert all((k == 'lr-ASGD-momentum' for k in lr_monitor.last_momentum_values))

def test_lr_monitor_no_logger(tmpdir):
    if False:
        while True:
            i = 10
    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=[lr_monitor], logger=False)
    with pytest.raises(MisconfigurationException, match='`Trainer` that has no logger'):
        trainer.fit(model)

@pytest.mark.parametrize('logging_interval', ['step', 'epoch'])
def test_lr_monitor_multi_lrs(tmpdir, logging_interval: str):
    if False:
        print('Hello World!')
    'Test that learning rates are extracted and logged for multi lr schedulers.'

    class CustomBoringModel(BoringModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            if False:
                return 10
            (opt1, opt2) = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            loss = self.loss(self.step(batch))
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

        def on_train_epoch_end(self):
            if False:
                i = 10
                return i + 15
            (scheduler1, scheduler2) = self.lr_schedulers()
            scheduler1.step()
            scheduler2.step()

        def configure_optimizers(self):
            if False:
                print('Hello World!')
            optimizer1 = optim.Adam(self.parameters(), lr=0.01)
            optimizer2 = optim.Adam(self.parameters(), lr=0.01)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
            return ([optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2])
    model = CustomBoringModel()
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    log_every_n_steps = 2
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, log_every_n_steps=log_every_n_steps, limit_train_batches=7, limit_val_batches=0.1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ['lr-Adam', 'lr-Adam-1'], 'Names of learning rates not set correctly'
    if logging_interval == 'step':
        expected_number_logged = trainer.global_step // 2 // log_every_n_steps
    if logging_interval == 'epoch':
        expected_number_logged = trainer.max_epochs
    assert all((len(lr) == expected_number_logged for lr in lr_monitor.lrs.values()))

@pytest.mark.parametrize('logging_interval', ['step', 'epoch'])
def test_lr_monitor_no_lr_scheduler_multi_lrs(tmpdir, logging_interval: str):
    if False:
        while True:
            i = 10
    'Test that learning rates are extracted and logged for multi optimizers but no lr scheduler.'

    class CustomBoringModel(BoringModel):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            if False:
                while True:
                    i = 10
            (opt1, opt2) = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            loss = self.loss(self.step(batch))
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            optimizer1 = optim.Adam(self.parameters(), lr=0.01)
            optimizer2 = optim.Adam(self.parameters(), lr=0.01)
            return [optimizer1, optimizer2]
    model = CustomBoringModel()
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    log_every_n_steps = 2
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, log_every_n_steps=log_every_n_steps, limit_train_batches=7, limit_val_batches=0.1, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model)
    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == len(trainer.optimizers)
    assert list(lr_monitor.lrs) == ['lr-Adam', 'lr-Adam-1'], 'Names of learning rates not set correctly'
    if logging_interval == 'step':
        expected_number_logged = trainer.global_step // 2 // log_every_n_steps
    if logging_interval == 'epoch':
        expected_number_logged = trainer.max_epochs
    assert all((len(lr) == expected_number_logged for lr in lr_monitor.lrs.values()))

@RunIf(sklearn=True)
def test_lr_monitor_param_groups(tmpdir):
    if False:
        while True:
            i = 10
    'Test that learning rates are extracted and logged for single lr scheduler.'

    class CustomClassificationModel(ClassificationModel):

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            param_groups = [{'params': list(self.parameters())[:2], 'lr': self.lr * 0.1}, {'params': list(self.parameters())[2:], 'lr': self.lr}]
            optimizer = optim.Adam(param_groups)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
            return ([optimizer], [lr_scheduler])
    model = CustomClassificationModel()
    dm = ClassifDataModule()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor], logger=CSVLogger(tmpdir))
    trainer.fit(model, datamodule=dm)
    assert lr_monitor.lrs, 'No learning rates logged'
    assert len(lr_monitor.lrs) == 2 * len(trainer.lr_scheduler_configs)
    assert list(lr_monitor.lrs) == ['lr-Adam/pg1', 'lr-Adam/pg2'], 'Names of learning rates not set correctly'

def test_lr_monitor_custom_name(tmpdir):
    if False:
        i = 10
        return i + 15

    class TestModel(BoringModel):

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            (optimizer, [scheduler]) = super().configure_optimizers()
            lr_scheduler = {'scheduler': scheduler, 'name': 'my_logging_name'}
            return (optimizer, [lr_scheduler])
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=0.1, limit_train_batches=0.5, callbacks=[lr_monitor], enable_progress_bar=False, enable_model_summary=False, logger=CSVLogger(tmpdir))
    trainer.fit(TestModel())
    assert list(lr_monitor.lrs) == ['my_logging_name']

def test_lr_monitor_custom_pg_name(tmpdir):
    if False:
        print('Hello World!')

    class TestModel(BoringModel):

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            optimizer = torch.optim.SGD([{'params': list(self.layer.parameters()), 'name': 'linear'}], lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return ([optimizer], [lr_scheduler])
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=2, limit_train_batches=2, callbacks=[lr_monitor], logger=CSVLogger(tmpdir), enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(TestModel())
    assert list(lr_monitor.lrs) == ['lr-SGD/linear']

def test_lr_monitor_duplicate_custom_pg_names(tmpdir):
    if False:
        i = 10
        return i + 15

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.linear_a = torch.nn.Linear(32, 16)
            self.linear_b = torch.nn.Linear(16, 2)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            param_groups = [{'params': list(self.linear_a.parameters()), 'name': 'linear'}, {'params': list(self.linear_b.parameters()), 'name': 'linear'}]
            optimizer = torch.optim.SGD(param_groups, lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return ([optimizer], [lr_scheduler])
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=2, limit_train_batches=2, callbacks=[lr_monitor], logger=CSVLogger(tmpdir), enable_progress_bar=False, enable_model_summary=False)
    with pytest.raises(MisconfigurationException, match='A single `Optimizer` cannot have multiple parameter groups with identical'):
        trainer.fit(TestModel())

def test_multiple_optimizers_basefinetuning(tmpdir):
    if False:
        for i in range(10):
            print('nop')

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.automatic_optimization = False
            self.backbone = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 32), torch.nn.Linear(32, 32), torch.nn.ReLU(True))
            self.layer = torch.nn.Linear(32, 2)

        def training_step(self, batch, batch_idx):
            if False:
                i = 10
                return i + 15
            (opt1, opt2, opt3) = self.optimizers()
            loss = self.step(batch)
            self.manual_backward(loss)
            opt1.step()
            opt1.zero_grad()
            loss = self.step(batch)
            self.manual_backward(loss)
            opt2.step()
            opt2.zero_grad()
            loss = self.step(batch)
            self.manual_backward(loss)
            opt3.step()
            opt3.zero_grad()

        def on_train_epoch_end(self) -> None:
            if False:
                i = 10
                return i + 15
            (lr_sched1, lr_sched2) = self.lr_schedulers()
            lr_sched1.step()
            lr_sched2.step()

        def forward(self, x):
            if False:
                return 10
            return self.layer(self.backbone(x))

        def configure_optimizers(self):
            if False:
                i = 10
                return i + 15
            parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
            opt = optim.SGD(parameters, lr=0.1)
            opt_2 = optim.Adam(parameters, lr=0.1)
            opt_3 = optim.AdamW(parameters, lr=0.1)
            optimizers = [opt, opt_2, opt_3]
            schedulers = [optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5), optim.lr_scheduler.StepLR(opt_2, step_size=1, gamma=0.5)]
            return (optimizers, schedulers)

    class Check(Callback):

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if False:
                print('Hello World!')
            num_param_groups = sum((len(opt.param_groups) for opt in trainer.optimizers))
            if trainer.current_epoch == 0:
                assert num_param_groups == 3
            elif trainer.current_epoch == 1:
                assert num_param_groups == 4
                assert list(lr_monitor.lrs) == ['lr-Adam', 'lr-AdamW', 'lr-SGD/pg1', 'lr-SGD/pg2']
            elif trainer.current_epoch == 2:
                assert num_param_groups == 5
                assert list(lr_monitor.lrs) == ['lr-AdamW', 'lr-SGD/pg1', 'lr-SGD/pg2', 'lr-Adam/pg1', 'lr-Adam/pg2']
            else:
                expected = ['lr-AdamW', 'lr-SGD/pg1', 'lr-SGD/pg2', 'lr-Adam/pg1', 'lr-Adam/pg2', 'lr-Adam/pg3']
                assert list(lr_monitor.lrs) == expected

    class TestFinetuning(BackboneFinetuning):

        def freeze_before_training(self, pl_module):
            if False:
                i = 10
                return i + 15
            self.freeze(pl_module.backbone[0])
            self.freeze(pl_module.backbone[1])
            self.freeze(pl_module.layer)

        def finetune_function(self, pl_module, epoch: int, optimizer):
            if False:
                while True:
                    i = 10
            'Called when the epoch begins.'
            if epoch == 1 and isinstance(optimizer, torch.optim.SGD):
                self.unfreeze_and_add_param_group(pl_module.backbone[0], optimizer, lr=0.1)
            if epoch == 2 and isinstance(optimizer, torch.optim.Adam):
                self.unfreeze_and_add_param_group(pl_module.layer, optimizer, lr=0.1)
            if epoch == 3 and isinstance(optimizer, torch.optim.Adam):
                assert len(optimizer.param_groups) == 2
                self.unfreeze_and_add_param_group(pl_module.backbone[1], optimizer, lr=0.1)
                assert len(optimizer.param_groups) == 3
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=5, limit_val_batches=0, limit_train_batches=2, callbacks=[TestFinetuning(), lr_monitor, Check()], logger=CSVLogger(tmpdir), enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False)
    model = TestModel()
    trainer.fit(model)
    expected = [0.1, 0.1, 0.1, 0.1, 0.1]
    assert lr_monitor.lrs['lr-AdamW'] == expected
    expected = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    assert lr_monitor.lrs['lr-SGD/pg1'] == expected
    expected = [0.1, 0.05, 0.025, 0.0125]
    assert lr_monitor.lrs['lr-SGD/pg2'] == expected
    expected = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    assert lr_monitor.lrs['lr-Adam/pg1'] == expected
    expected = [0.1, 0.05, 0.025]
    assert lr_monitor.lrs['lr-Adam/pg2'] == expected
    expected = [0.1, 0.05]
    assert lr_monitor.lrs['lr-Adam/pg3'] == expected

def test_lr_monitor_multiple_param_groups_no_lr_scheduler(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that the `LearningRateMonitor` is able to log correct keys with multiple param groups and no\n    lr_scheduler.'

    class TestModel(BoringModel):

        def __init__(self, lr, momentum):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.save_hyperparameters()
            self.linear_a = torch.nn.Linear(32, 16)
            self.linear_b = torch.nn.Linear(16, 2)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            param_groups = [{'params': list(self.linear_a.parameters()), 'weight_decay': 0.1}, {'params': list(self.linear_b.parameters()), 'weight_decay': 0.1}]
            return torch.optim.Adam(param_groups, lr=self.hparams.lr, betas=self.hparams.momentum)
    lr_monitor = LearningRateMonitor(log_momentum=True, log_weight_decay=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_val_batches=2, limit_train_batches=2, callbacks=[lr_monitor], logger=CSVLogger(tmpdir), enable_progress_bar=False, enable_model_summary=False)
    lr = 0.01
    momentum = 0.7
    weight_decay = 0.1
    model = TestModel(lr=lr, momentum=(momentum, 0.999))
    trainer.fit(model)
    assert len(lr_monitor.lrs) == len(trainer.optimizers[0].param_groups)
    assert list(lr_monitor.lrs) == ['lr-Adam/pg1', 'lr-Adam/pg2']
    assert list(lr_monitor.last_momentum_values) == ['lr-Adam/pg1-momentum', 'lr-Adam/pg2-momentum']
    assert all((val == momentum for val in lr_monitor.last_momentum_values.values()))
    assert list(lr_monitor.last_weight_decay_values) == ['lr-Adam/pg1-weight_decay', 'lr-Adam/pg2-weight_decay']
    assert all((val == weight_decay for val in lr_monitor.last_weight_decay_values.values()))
    assert all((all((val == lr for val in lr_monitor.lrs[lr_key])) for lr_key in lr_monitor.lrs))

def test_lr_monitor_update_callback_metrics(tmpdir):
    if False:
        return 10
    'Test that the `LearningRateMonitor` callback updates trainer.callback_metrics.'

    class TestModel(BoringModel):

        def configure_optimizers(self):
            if False:
                print('Hello World!')
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            return ([optimizer], [lr_scheduler])
    monitor_key = 'lr-SGD'
    stop_threshold = 0.02
    expected_stop_epoch = 3
    lr_monitor = LearningRateMonitor()
    lr_es = EarlyStopping(monitor=monitor_key, mode='min', stopping_threshold=stop_threshold, check_on_train_epoch_end=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[lr_monitor, lr_es], max_epochs=5, limit_val_batches=0, limit_train_batches=2, logger=CSVLogger(tmpdir))
    model = TestModel()
    trainer.fit(model)
    assert monitor_key in trainer.callback_metrics
    assert lr_monitor.lrs[monitor_key] == [0.1, 0.05, 0.025, 0.0125]
    assert min(lr_monitor.lrs[monitor_key][:expected_stop_epoch]) > stop_threshold
    assert len(lr_monitor.lrs[monitor_key][expected_stop_epoch:]) == 1
    assert min(lr_monitor.lrs[monitor_key][expected_stop_epoch:]) < stop_threshold
    assert trainer.current_epoch - 1 == expected_stop_epoch
    assert lr_es.stopped_epoch == expected_stop_epoch