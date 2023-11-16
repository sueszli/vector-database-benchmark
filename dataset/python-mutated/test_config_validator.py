from unittest.mock import Mock
import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.trainer.configuration_validator import __verify_eval_loop_configuration, __verify_train_val_loop_configuration
from lightning.pytorch.utilities.exceptions import MisconfigurationException

def test_wrong_train_setting(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that an error is raised when no `training_step()` is defined.'
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    with pytest.raises(MisconfigurationException, match='No `training_step\\(\\)` method defined.'):
        model = BoringModel()
        model.training_step = None
        trainer.fit(model)

def test_wrong_configure_optimizers(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that an error is thrown when no `configure_optimizers()` is defined.'
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    with pytest.raises(MisconfigurationException, match='No `configure_optimizers\\(\\)` method defined.'):
        model = BoringModel()
        model.configure_optimizers = None
        trainer.fit(model)

def test_fit_val_loop_config(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'When either val loop or val data are missing raise warning.'
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    with pytest.warns(UserWarning, match='You passed in a `val_dataloader` but have no `validation_step`'):
        model = BoringModel()
        model.validation_step = None
        trainer.fit(model)
    with pytest.warns(PossibleUserWarning, match='You defined a `validation_step` but have no `val_dataloader`'):
        model = BoringModel()
        model.val_dataloader = None
        trainer.fit(model)

def test_eval_loop_config(tmpdir):
    if False:
        i = 10
        return i + 15
    'When either eval step or eval data is missing.'
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    model = BoringModel()
    model.validation_step = None
    with pytest.raises(MisconfigurationException, match='No `validation_step\\(\\)` method defined'):
        trainer.validate(model)
    model = BoringModel()
    model.test_step = None
    with pytest.raises(MisconfigurationException, match='No `test_step\\(\\)` method defined'):
        trainer.test(model)
    model = BoringModel()
    model.predict_step = None
    with pytest.raises(MisconfigurationException, match='`predict_step` cannot be None.'):
        trainer.predict(model)
    model = BoringModel()
    model.forward = None
    with pytest.raises(MisconfigurationException, match='requires `forward` method to run.'):
        trainer.predict(model)

@pytest.mark.parametrize('datamodule', [False, True])
def test_trainer_predict_verify_config(tmpdir, datamodule):
    if False:
        print('Hello World!')

    class TestModel(LightningModule):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            return self.layer(x)

    class TestLightningDataModule(LightningDataModule):

        def __init__(self, dataloaders):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self._dataloaders = dataloaders

        def test_dataloader(self):
            if False:
                return 10
            return self._dataloaders

        def predict_dataloader(self):
            if False:
                print('Hello World!')
            return self._dataloaders
    data = [torch.utils.data.DataLoader(RandomDataset(32, 2)), torch.utils.data.DataLoader(RandomDataset(32, 2))]
    if datamodule:
        data = TestLightningDataModule(data)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir)
    results = trainer.predict(model, data)
    assert len(results) == 2
    assert results[0][0].shape == torch.Size([1, 2])

def test_trainer_manual_optimization_config():
    if False:
        print('Hello World!')
    'Test error message when requesting Trainer features unsupported with manual optimization.'
    model = BoringModel()
    model.automatic_optimization = False
    trainer = Trainer(gradient_clip_val=1.0)
    with pytest.raises(MisconfigurationException, match='Automatic gradient clipping is not supported'):
        trainer.fit(model)
    trainer = Trainer(accumulate_grad_batches=2)
    with pytest.raises(MisconfigurationException, match='Automatic gradient accumulation is not supported'):
        trainer.fit(model)

def test_legacy_epoch_end_hooks():
    if False:
        i = 10
        return i + 15

    class TrainingEpochEndModel(BoringModel):

        def training_epoch_end(self, outputs):
            if False:
                i = 10
                return i + 15
            pass

    class ValidationEpochEndModel(BoringModel):

        def validation_epoch_end(self, outputs):
            if False:
                i = 10
                return i + 15
            pass
    trainer = Mock()
    with pytest.raises(NotImplementedError, match='training_epoch_end` has been removed in v2.0'):
        __verify_train_val_loop_configuration(trainer, TrainingEpochEndModel())
    with pytest.raises(NotImplementedError, match='validation_epoch_end` has been removed in v2.0'):
        __verify_train_val_loop_configuration(trainer, ValidationEpochEndModel())

    class TestEpochEndModel(BoringModel):

        def test_epoch_end(self, outputs):
            if False:
                i = 10
                return i + 15
            pass
    with pytest.raises(NotImplementedError, match='validation_epoch_end` has been removed in v2.0'):
        __verify_eval_loop_configuration(ValidationEpochEndModel(), 'val')
    with pytest.raises(NotImplementedError, match='test_epoch_end` has been removed in v2.0'):
        __verify_eval_loop_configuration(TestEpochEndModel(), 'test')