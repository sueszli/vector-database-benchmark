import contextlib
import logging
from unittest import mock
from unittest.mock import Mock
import pytest
import torch
from lightning.fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_8_0, _PYTHON_GREATER_EQUAL_3_10_0
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, GradientAccumulationScheduler, LearningRateMonitor, ModelCheckpoint, ModelSummary, ProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector

def test_checkpoint_callbacks_are_last(tmpdir):
    if False:
        print('Hello World!')
    'Test that checkpoint callbacks always get moved to the end of the list, with preserved order.'
    checkpoint1 = ModelCheckpoint(tmpdir, monitor='foo')
    checkpoint2 = ModelCheckpoint(tmpdir, monitor='bar')
    model_summary = ModelSummary()
    early_stopping = EarlyStopping(monitor='foo')
    lr_monitor = LearningRateMonitor()
    progress_bar = TQDMProgressBar()
    trainer = Trainer(callbacks=[checkpoint1, progress_bar, lr_monitor, model_summary, checkpoint2])
    assert trainer.callbacks == [progress_bar, lr_monitor, model_summary, checkpoint1, checkpoint2]
    model = LightningModule()
    model.configure_callbacks = lambda : []
    trainer.strategy._lightning_module = model
    cb_connector = _CallbackConnector(trainer)
    cb_connector._attach_model_callbacks()
    assert trainer.callbacks == [progress_bar, lr_monitor, model_summary, checkpoint1, checkpoint2]
    model = LightningModule()
    model.configure_callbacks = lambda : [checkpoint1, early_stopping, model_summary, checkpoint2]
    trainer = Trainer(callbacks=[progress_bar, lr_monitor, ModelCheckpoint(tmpdir)])
    trainer.strategy._lightning_module = model
    cb_connector = _CallbackConnector(trainer)
    cb_connector._attach_model_callbacks()
    assert trainer.callbacks == [progress_bar, lr_monitor, early_stopping, model_summary, checkpoint1, checkpoint2]
    model = LightningModule()
    batch_size_finder = BatchSizeFinder()
    model.configure_callbacks = lambda : [checkpoint2, early_stopping, batch_size_finder, model_summary, checkpoint1]
    trainer = Trainer(callbacks=[progress_bar, lr_monitor])
    trainer.strategy._lightning_module = model
    cb_connector = _CallbackConnector(trainer)
    cb_connector._attach_model_callbacks()
    assert trainer.callbacks == [batch_size_finder, progress_bar, lr_monitor, early_stopping, model_summary, checkpoint2, checkpoint1]

class StatefulCallback0(Callback):

    def state_dict(self):
        if False:
            i = 10
            return i + 15
        return {'content0': 0}

class StatefulCallback1(Callback):

    def __init__(self, unique=None, other=None):
        if False:
            for i in range(10):
                print('nop')
        self._unique = unique
        self._other = other

    @property
    def state_key(self):
        if False:
            while True:
                i = 10
        return self._generate_state_key(unique=self._unique)

    def state_dict(self):
        if False:
            while True:
                i = 10
        return {'content1': self._unique}

def test_all_callback_states_saved_before_checkpoint_callback(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that all callback states get saved even if the ModelCheckpoint is not given as last and when there are\n    multiple callbacks of the same type.'
    callback0 = StatefulCallback0()
    callback1 = StatefulCallback1(unique='one')
    callback2 = StatefulCallback1(unique='two', other=2)
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='all_states')
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, limit_val_batches=1, callbacks=[callback0, checkpoint_callback, callback1, callback2])
    trainer.fit(model)
    ckpt = torch.load(str(tmpdir / 'all_states.ckpt'))
    state0 = ckpt['callbacks']['StatefulCallback0']
    state1 = ckpt['callbacks']["StatefulCallback1{'unique': 'one'}"]
    state2 = ckpt['callbacks']["StatefulCallback1{'unique': 'two'}"]
    assert 'content0' in state0
    assert state0['content0'] == 0
    assert 'content1' in state1
    assert state1['content1'] == 'one'
    assert 'content1' in state2
    assert state2['content1'] == 'two'
    assert "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}" in ckpt['callbacks']

def test_attach_model_callbacks():
    if False:
        i = 10
        return i + 15
    'Test that the callbacks defined in the model and through Trainer get merged correctly.'

    def _attach_callbacks(trainer_callbacks, model_callbacks):
        if False:
            i = 10
            return i + 15
        model = LightningModule()
        model.configure_callbacks = lambda : model_callbacks
        has_progress_bar = any((isinstance(cb, ProgressBar) for cb in trainer_callbacks + model_callbacks))
        trainer = Trainer(enable_checkpointing=False, enable_progress_bar=has_progress_bar, enable_model_summary=False, callbacks=trainer_callbacks)
        trainer.strategy._lightning_module = model
        cb_connector = _CallbackConnector(trainer)
        cb_connector._attach_model_callbacks()
        return trainer
    early_stopping1 = EarlyStopping(monitor='red')
    early_stopping2 = EarlyStopping(monitor='blue')
    progress_bar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor()
    grad_accumulation = GradientAccumulationScheduler({1: 1})
    trainer = _attach_callbacks(trainer_callbacks=[], model_callbacks=[])
    assert trainer.callbacks == []
    trainer = _attach_callbacks(trainer_callbacks=[early_stopping1], model_callbacks=[progress_bar])
    assert trainer.callbacks == [early_stopping1, progress_bar]
    trainer = _attach_callbacks(trainer_callbacks=[progress_bar, EarlyStopping(monitor='red')], model_callbacks=[early_stopping1])
    assert trainer.callbacks == [progress_bar, early_stopping1]
    trainer = _attach_callbacks(trainer_callbacks=[LearningRateMonitor(), EarlyStopping(monitor='yellow'), LearningRateMonitor(), EarlyStopping(monitor='black')], model_callbacks=[early_stopping1, lr_monitor])
    assert trainer.callbacks == [early_stopping1, lr_monitor]
    trainer = _attach_callbacks(trainer_callbacks=[LearningRateMonitor(), progress_bar, EarlyStopping(monitor='yellow'), LearningRateMonitor(), EarlyStopping(monitor='black')], model_callbacks=[early_stopping1, lr_monitor, grad_accumulation, early_stopping2])
    assert trainer.callbacks == [progress_bar, early_stopping1, lr_monitor, grad_accumulation, early_stopping2]

    class CustomProgressBar(TQDMProgressBar):
        ...
    custom_progress_bar = CustomProgressBar()
    trainer = _attach_callbacks(trainer_callbacks=[progress_bar], model_callbacks=[custom_progress_bar])
    assert trainer.callbacks == [custom_progress_bar]
    bare_callback = Callback()
    trainer = _attach_callbacks(trainer_callbacks=[bare_callback], model_callbacks=[custom_progress_bar])
    assert trainer.callbacks == [bare_callback, custom_progress_bar]

def test_attach_model_callbacks_override_info(caplog):
    if False:
        while True:
            i = 10
    'Test that the logs contain the info about overriding callbacks returned by configure_callbacks.'
    model = LightningModule()
    model.configure_callbacks = lambda : [LearningRateMonitor(), EarlyStopping(monitor='foo')]
    trainer = Trainer(enable_checkpointing=False, callbacks=[EarlyStopping(monitor='foo'), LearningRateMonitor(), TQDMProgressBar()])
    trainer.strategy._lightning_module = model
    cb_connector = _CallbackConnector(trainer)
    with caplog.at_level(logging.INFO):
        cb_connector._attach_model_callbacks()
    assert 'existing callbacks passed to Trainer: EarlyStopping, LearningRateMonitor' in caplog.text

class ExternalCallback(Callback):
    """A callback in another library that gets registered through entry points."""
    pass

def test_configure_external_callbacks():
    if False:
        return 10
    'Test that the connector collects Callback instances from factories registered through entry points.'

    def factory_no_callback():
        if False:
            while True:
                i = 10
        return []

    def factory_one_callback():
        if False:
            i = 10
            return i + 15
        return ExternalCallback()

    def factory_one_callback_list():
        if False:
            i = 10
            return i + 15
        return [ExternalCallback()]

    def factory_multiple_callbacks_list():
        if False:
            i = 10
            return i + 15
        return [ExternalCallback(), ExternalCallback()]
    with _make_entry_point_query_mock(factory_no_callback):
        trainer = Trainer(enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert trainer.callbacks == []
    with _make_entry_point_query_mock(factory_one_callback):
        trainer = Trainer(enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert isinstance(trainer.callbacks[0], ExternalCallback)
    with _make_entry_point_query_mock(factory_one_callback_list):
        trainer = Trainer(enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert isinstance(trainer.callbacks[0], ExternalCallback)
    with _make_entry_point_query_mock(factory_multiple_callbacks_list):
        trainer = Trainer(enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert isinstance(trainer.callbacks[0], ExternalCallback)
    assert isinstance(trainer.callbacks[1], ExternalCallback)

@contextlib.contextmanager
def _make_entry_point_query_mock(callback_factory):
    if False:
        return 10
    query_mock = Mock()
    entry_point = Mock()
    entry_point.name = 'mocked'
    entry_point.load.return_value = callback_factory
    if _PYTHON_GREATER_EQUAL_3_10_0:
        query_mock.return_value = [entry_point]
        import_path = 'importlib.metadata.entry_points'
    elif _PYTHON_GREATER_EQUAL_3_8_0:
        query_mock().get.return_value = [entry_point]
        import_path = 'importlib.metadata.entry_points'
    else:
        query_mock.return_value = [entry_point]
        import_path = 'pkg_resources.iter_entry_points'
    with mock.patch(import_path, query_mock):
        yield

def test_validate_unique_callback_state_key():
    if False:
        print('Hello World!')
    'Test that we raise an error if the state keys collide, leading to missing state in the checkpoint.'

    class MockCallback(Callback):

        @property
        def state_key(self):
            if False:
                return 10
            return 'same_key'

        def state_dict(self):
            if False:
                return 10
            return {'state': 1}
    with pytest.raises(RuntimeError, match='Found more than one stateful callback of type `MockCallback`'):
        Trainer(callbacks=[MockCallback(), MockCallback()])