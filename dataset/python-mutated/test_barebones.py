import logging
import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.profilers import PassThroughProfiler

def test_barebones_disables_logging():
    if False:
        while True:
            i = 10
    pl_module = BoringModel()
    trainer = Trainer(barebones=True)
    pl_module._trainer = trainer
    with pytest.warns(match='barebones=True\\)` is configured'):
        pl_module.log('foo', 1.0)
    with pytest.warns(match='barebones=True\\)` is configured'):
        pl_module.log_dict({'foo': 1.0})

def test_barebones_argument_selection(caplog):
    if False:
        return 10
    with caplog.at_level(logging.INFO):
        trainer = Trainer(barebones=True)
    assert 'running in `Trainer(barebones=True)` mode' in caplog.text
    assert trainer.barebones
    assert not trainer.checkpoint_callbacks
    assert not trainer.loggers
    assert not trainer.progress_bar_callback
    assert not any((isinstance(cb, ModelSummary) for cb in trainer.callbacks))
    assert not trainer.log_every_n_steps
    assert not trainer.num_sanity_val_steps
    assert not trainer.fast_dev_run
    assert not trainer._detect_anomaly
    assert isinstance(trainer.profiler, PassThroughProfiler)

def test_barebones_raises():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='enable_checkpointing=True\\)` was passed'):
        Trainer(barebones=True, enable_checkpointing=True)
    with pytest.raises(ValueError, match='logger=True\\)` was passed'):
        Trainer(barebones=True, logger=True)
    with pytest.raises(ValueError, match='enable_progress_bar=True\\)` was passed'):
        Trainer(barebones=True, enable_progress_bar=True)
    with pytest.raises(ValueError, match='enable_model_summary=True\\)` was passed'):
        Trainer(barebones=True, enable_model_summary=True)
    with pytest.raises(ValueError, match='log_every_n_steps=1\\)` was passed'):
        Trainer(barebones=True, log_every_n_steps=1)
    with pytest.raises(ValueError, match='num_sanity_val_steps=1\\)` was passed'):
        Trainer(barebones=True, num_sanity_val_steps=1)
    with pytest.raises(ValueError, match='fast_dev_run=1\\)` was passed'):
        Trainer(barebones=True, fast_dev_run=1)
    with pytest.raises(ValueError, match='detect_anomaly=True\\)` was passed'):
        Trainer(barebones=True, detect_anomaly=True)
    with pytest.raises(ValueError, match="profiler='simple'\\)` was passed"):
        Trainer(barebones=True, profiler='simple')