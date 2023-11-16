import logging
import pytest
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

@pytest.mark.parametrize('max_epochs', [1, 2, 3])
@pytest.mark.parametrize('denominator', [1, 3, 4])
def test_val_check_interval(tmpdir, max_epochs, denominator):
    if False:
        while True:
            i = 10

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            if False:
                while True:
                    i = 10
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if False:
                while True:
                    i = 10
            if not self.trainer.sanity_checking:
                self.val_epoch_calls += 1
    model = TestModel()
    trainer = Trainer(max_epochs=max_epochs, val_check_interval=1 / denominator, logger=False)
    trainer.fit(model)
    assert model.train_epoch_calls == max_epochs
    assert model.val_epoch_calls == max_epochs * denominator

@pytest.mark.parametrize('value', [1, 1.0])
def test_val_check_interval_info_message(caplog, value):
    if False:
        return 10
    with caplog.at_level(logging.INFO):
        Trainer(val_check_interval=value)
    assert f'`Trainer(val_check_interval={value})` was configured' in caplog.text
    message = 'configured so validation will run'
    assert message in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        Trainer()
    assert message not in caplog.text

@pytest.mark.parametrize('use_infinite_dataset', [True, False])
@pytest.mark.parametrize('accumulate_grad_batches', [1, 2])
def test_validation_check_interval_exceed_data_length_correct(tmpdir, use_infinite_dataset, accumulate_grad_batches):
    if False:
        while True:
            i = 10
    data_samples_train = 4
    max_epochs = 3
    max_steps = data_samples_train * max_epochs
    max_opt_steps = max_steps // accumulate_grad_batches

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.validation_called_at_step = set()

        def validation_step(self, *args):
            if False:
                while True:
                    i = 10
            self.validation_called_at_step.add(self.trainer.fit_loop.total_batch_idx + 1)
            return super().validation_step(*args)

        def train_dataloader(self):
            if False:
                print('Hello World!')
            train_ds = RandomIterableDataset(32, count=max_steps + 100) if use_infinite_dataset else RandomDataset(32, length=data_samples_train)
            return DataLoader(train_ds)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_val_batches=1, max_steps=max_opt_steps, val_check_interval=3, check_val_every_n_epoch=None, num_sanity_val_steps=0, accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(model)
    assert trainer.current_epoch == 1 if use_infinite_dataset else max_epochs
    assert trainer.global_step == max_opt_steps
    assert sorted(model.validation_called_at_step) == [3, 6, 9, 12]

def test_validation_check_interval_exceed_data_length_wrong():
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(limit_train_batches=10, val_check_interval=100)
    model = BoringModel()
    with pytest.raises(ValueError, match='must be less than or equal to the number of the training batches'):
        trainer.fit(model)

def test_val_check_interval_float_with_none_check_val_every_n_epoch():
    if False:
        i = 10
        return i + 15
    'Test that an exception is raised when `val_check_interval` is set to float with\n    `check_val_every_n_epoch=None`'
    with pytest.raises(MisconfigurationException, match='`val_check_interval` should be an integer when `check_val_every_n_epoch=None`'):
        Trainer(val_check_interval=0.5, check_val_every_n_epoch=None)