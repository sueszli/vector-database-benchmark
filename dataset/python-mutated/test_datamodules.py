import pickle
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict
from unittest import mock
from unittest.mock import Mock, PropertyMock, call
import pytest
import torch
from lightning.pytorch import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.profilers.simple import SimpleProfiler
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import AttributeDict
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf

@mock.patch('lightning.pytorch.trainer.trainer.Trainer.node_rank', new_callable=PropertyMock)
@mock.patch('lightning.pytorch.trainer.trainer.Trainer.local_rank', new_callable=PropertyMock)
def test_can_prepare_data(local_rank, node_rank):
    if False:
        return 10
    dm = Mock(spec=LightningDataModule)
    dm.prepare_data_per_node = True
    trainer = Trainer()
    trainer.datamodule = dm
    dm.prepare_data.assert_not_called()
    local_rank.return_value = 0
    assert trainer.local_rank == 0
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()
    dm.reset_mock()
    local_rank.return_value = 1
    assert trainer.local_rank == 1
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()
    dm.reset_mock()
    dm.prepare_data_per_node = False
    node_rank.return_value = 0
    local_rank.return_value = 0
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()
    dm.reset_mock()
    node_rank.return_value = 1
    local_rank.return_value = 0
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()
    node_rank.return_value = 0
    local_rank.return_value = 1
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_not_called()
    dm.prepare_data_per_node = True
    local_rank.return_value = 0
    trainer._data_connector.prepare_data()
    dm.prepare_data.assert_called_once()

def test_hooks_no_recursion_error():
    if False:
        print('Hello World!')

    class DummyDM(LightningDataModule):

        def setup(self, *args, **kwargs):
            if False:
                return 10
            pass

        def prepare_data(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            pass
    for i in range(1005):
        dm = DummyDM()
        dm.setup()
        dm.prepare_data()

def test_helper_boringdatamodule():
    if False:
        return 10
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup('fit')

def test_helper_boringdatamodule_with_verbose_setup():
    if False:
        while True:
            i = 10
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')

class DataDirDataModule(BoringDataModule):

    def __init__(self, data_dir: str):
        if False:
            return 10
        super().__init__()
        self.data_dir = data_dir

def test_dm_pickle_after_init():
    if False:
        for i in range(10):
            print('nop')
    dm = BoringDataModule()
    pickle.dumps(dm)

@RunIf(sklearn=True)
def test_train_loop_only(tmpdir):
    if False:
        while True:
            i = 10
    seed_everything(7)
    dm = ClassifDataModule()
    model = ClassificationModel()
    model.validation_step = None
    model.test_step = None
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f'Training failed with {trainer.state}'
    assert trainer.callback_metrics['train_loss'] < 1.1

@RunIf(sklearn=True)
def test_train_val_loop_only(tmpdir):
    if False:
        return 10
    seed_everything(7)
    dm = ClassifDataModule()
    model = ClassificationModel()
    model.validation_step = None
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f'Training failed with {trainer.state}'
    assert trainer.callback_metrics['train_loss'] < 1.1

def test_dm_checkpoint_save_and_load(tmpdir):
    if False:
        i = 10
        return i + 15

    class CustomBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            if False:
                i = 10
                return i + 15
            out = super().validation_step(batch, batch_idx)
            self.log('early_stop_on', out['x'])
            return out

    class CustomBoringDataModule(BoringDataModule):

        def state_dict(self) -> Dict[str, Any]:
            if False:
                while True:
                    i = 10
            return {'my': 'state_dict'}

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            if False:
                print('Hello World!')
            self.my_state_dict = state_dict
    dm = CustomBoringDataModule()
    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, limit_val_batches=1, enable_model_summary=False, callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on')])
    trainer.fit(model, datamodule=dm)
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__qualname__ in checkpoint
    assert checkpoint[dm.__class__.__qualname__] == {'my': 'state_dict'}
    for trainer_fn in TrainerFn:
        trainer.state.fn = trainer_fn
        trainer._checkpoint_connector._restore_modules_and_callbacks(checkpoint_path)
        assert dm.my_state_dict == {'my': 'state_dict'}

@RunIf(sklearn=True)
def test_full_loop(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    seed_everything(7)
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False, deterministic='warn')
    trainer.fit(model, dm)
    assert trainer.state.finished, f'Training failed with {trainer.state}'
    assert dm.trainer is not None
    result = trainer.validate(model, dm)
    assert dm.trainer is not None
    assert result[0]['val_acc'] > 0.6
    result = trainer.test(model, dm)
    assert dm.trainer is not None
    assert result[0]['test_acc'] > 0.57

def test_dm_reload_dataloaders_every_n_epochs(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test datamodule, where trainer argument reload_dataloaders_every_n_epochs is set to a non negative integer.'

    class CustomBoringDataModule(BoringDataModule):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self._epochs_called_for = []

        def train_dataloader(self):
            if False:
                return 10
            assert self.trainer.current_epoch not in self._epochs_called_for
            self._epochs_called_for.append(self.trainer.current_epoch)
            return super().train_dataloader()
    dm = CustomBoringDataModule()
    model = BoringModel()
    model.validation_step = None
    model.test_step = None
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, limit_train_batches=2, reload_dataloaders_every_n_epochs=2)
    trainer.fit(model, dm)

class DummyDS(torch.utils.data.Dataset):

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return 1

    def __len__(self):
        if False:
            return 10
        return 100

class DummyIDS(torch.utils.data.IterableDataset):

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield 1

@pytest.mark.parametrize('iterable', [False, True])
def test_dm_init_from_datasets_dataloaders(iterable):
    if False:
        i = 10
        return i + 15
    ds = DummyIDS if iterable else DummyDS
    train_ds = ds()
    dm = LightningDataModule.from_datasets(train_ds, batch_size=4, num_workers=0)
    with mock.patch('lightning.pytorch.core.datamodule.DataLoader') as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_called_once_with(train_ds, batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)
    with pytest.raises(MisconfigurationException, match='`val_dataloader` must be implemented'):
        _ = dm.val_dataloader()
    with pytest.raises(MisconfigurationException, match='`test_dataloader` must be implemented'):
        _ = dm.test_dataloader()
    train_ds_sequence = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds_sequence, batch_size=4, num_workers=0)
    with mock.patch('lightning.pytorch.core.datamodule.DataLoader') as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_has_calls([call(train_ds_sequence[0], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True), call(train_ds_sequence[1], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)])
    with pytest.raises(MisconfigurationException, match='`val_dataloader` must be implemented'):
        _ = dm.val_dataloader()
    with pytest.raises(MisconfigurationException, match='`test_dataloader` must be implemented'):
        _ = dm.test_dataloader()
    valid_ds = ds()
    test_ds = ds()
    dm = LightningDataModule.from_datasets(val_dataset=valid_ds, test_dataset=test_ds, batch_size=2, num_workers=0)
    with mock.patch('lightning.pytorch.core.datamodule.DataLoader') as dl_mock:
        dm.val_dataloader()
        dl_mock.assert_called_with(valid_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        dm.test_dataloader()
        dl_mock.assert_called_with(test_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    with pytest.raises(MisconfigurationException, match='`train_dataloader` must be implemented'):
        _ = dm.train_dataloader()
    valid_dss = [ds(), ds()]
    test_dss = [ds(), ds()]
    predict_dss = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds, valid_dss, test_dss, predict_dss, batch_size=4, num_workers=0)
    with mock.patch('lightning.pytorch.core.datamodule.DataLoader') as dl_mock:
        dm.val_dataloader()
        dm.test_dataloader()
        dm.predict_dataloader()
        dl_mock.assert_has_calls([call(valid_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True), call(valid_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True), call(test_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True), call(test_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True), call(predict_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True), call(predict_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True)])

def test_dm_init_from_datasets_with_init_params():
    if False:
        for i in range(10):
            print('nop')
    'Test that extra kwargs can be passed down to the init via the ``LightningDataModule.from_datasets`` method.\n\n    The two special arguments batch_size and num_workers get passed down depending on whether the __init__ accepts them.\n\n    '
    LightningDataModule.from_datasets(DummyDS(), batch_size=4, num_workers=2)

    class KnownExtraParametersDataModule(LightningDataModule):

        def __init__(self, batch_size=1, num_workers=0):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers
    dm = KnownExtraParametersDataModule.from_datasets(DummyDS(), batch_size=4, num_workers=2)
    assert dm.batch_size == 4
    assert dm.num_workers == 2

    class UnknownExtraParametersDataModule(LightningDataModule):

        def __init__(self, other, batch_size=1):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.other = other
            self.batch_size = batch_size
    dm = UnknownExtraParametersDataModule.from_datasets(DummyDS(), batch_size=4, num_workers=2, other=5)
    assert dm.batch_size == 4
    assert dm.other == 5
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'other'"):
        UnknownExtraParametersDataModule.from_datasets(DummyDS(), batch_size=4, num_workers=2)

    class KwargsParametersDataModule(LightningDataModule):

        def __init__(self, num_workers, **kwargs):
            if False:
                print('Hello World!')
            super().__init__()
            self.num_workers = num_workers
            for (key, value) in kwargs.items():
                setattr(self, key, value)
    dm = KwargsParametersDataModule.from_datasets(DummyDS(), batch_size=10, num_workers=100, another=None)
    assert dm.batch_size == 10
    assert dm.num_workers == 100
    assert dm.another is None

class DataModuleWithHparams_0(LightningDataModule):

    def __init__(self, arg0, arg1, kwarg0=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.save_hyperparameters()

class DataModuleWithHparams_1(LightningDataModule):

    def __init__(self, arg0, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.save_hyperparameters(arg0)

def test_hyperparameters_saving():
    if False:
        return 10
    data = DataModuleWithHparams_0(10, 'foo', kwarg0='bar')
    assert data.hparams == AttributeDict({'arg0': 10, 'arg1': 'foo', 'kwarg0': 'bar'})
    data = DataModuleWithHparams_1(Namespace(**{'hello': 'world'}), 'foo', kwarg0='bar')
    assert data.hparams == AttributeDict({'hello': 'world'})
    data = DataModuleWithHparams_1({'hello': 'world'}, 'foo', kwarg0='bar')
    assert data.hparams == AttributeDict({'hello': 'world'})
    if _OMEGACONF_AVAILABLE:
        data = DataModuleWithHparams_1(OmegaConf.create({'hello': 'world'}), 'foo', kwarg0='bar')
        assert data.hparams == OmegaConf.create({'hello': 'world'})

def test_define_as_dataclass():
    if False:
        i = 10
        return i + 15

    class BoringDataModule(LightningDataModule):

        def __init__(self, foo=None):
            if False:
                return 10
            super().__init__()

    @dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
    class BoringDataModule1(BoringDataModule):
        batch_size: int
        foo: int = 2

        def __post_init__(self):
            if False:
                return 10
            super().__init__(foo=self.foo)
    assert BoringDataModule1(batch_size=64).foo == 2
    assert BoringDataModule1(batch_size=32)
    assert hasattr(BoringDataModule1, '__repr__')
    assert BoringDataModule1(batch_size=32) == BoringDataModule1(batch_size=32)

    @dataclass
    class BoringDataModule2(LightningDataModule):
        batch_size: int
    assert BoringDataModule2(batch_size=32)
    assert hasattr(BoringDataModule2, '__repr__')
    assert BoringDataModule2(batch_size=32).prepare_data() is None
    assert BoringDataModule2(batch_size=32) == BoringDataModule2(batch_size=32)

@RunIf(skip_windows=True)
def test_datamodule_hooks_are_profiled():
    if False:
        while True:
            i = 10
    'Test that `LightningDataModule` hooks are profiled.'

    def get_trainer():
        if False:
            for i in range(10):
                print('nop')
        return Trainer(max_steps=1, limit_val_batches=0, profiler='simple', enable_model_summary=False, enable_progress_bar=False, logger=False)

    class CustomBoringDataModule(BoringDataModule):

        def state_dict(self):
            if False:
                i = 10
                return i + 15
            return {'temp': 1}
    model = BoringModel()
    dm = CustomBoringDataModule()
    trainer = get_trainer()
    trainer.fit(model, datamodule=dm)
    profiler = trainer.profiler
    assert isinstance(profiler, SimpleProfiler)
    keys = ['[LightningDataModule]CustomBoringDataModule.prepare_data', '[LightningDataModule]CustomBoringDataModule.setup', '[LightningDataModule]CustomBoringDataModule.state_dict', '[LightningDataModule]CustomBoringDataModule.teardown']
    for key in keys:
        assert key in profiler.recorded_durations
        durations = profiler.recorded_durations[key]
        assert len(durations) == 1
        assert durations[0] > 0
    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = get_trainer()
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    profiler = trainer.profiler
    keys = ['[LightningDataModule]CustomBoringDataModule.prepare_data', '[LightningDataModule]CustomBoringDataModule.setup', '[LightningDataModule]CustomBoringDataModule.load_state_dict', '[LightningDataModule]CustomBoringDataModule.teardown']
    for key in keys:
        assert key in profiler.recorded_durations
        durations = profiler.recorded_durations[key]
        assert len(durations) == 1
        assert durations[0] > 0