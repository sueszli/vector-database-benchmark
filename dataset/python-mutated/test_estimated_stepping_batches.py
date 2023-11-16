import logging
import os
from unittest import mock
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomIterableDataset
from lightning.pytorch.strategies import SingleDeviceXLAStrategy
from torch.utils.data import DataLoader
from tests_pytorch.conftest import mock_cuda_count
from tests_pytorch.helpers.runif import RunIf

def test_num_stepping_batches_basic():
    if False:
        print('Hello World!')
    'Test number of stepping batches in a general case.'
    max_epochs = 2
    trainer = Trainer(max_epochs=max_epochs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == 64 * max_epochs

def test_num_stepping_batches_raises_info_with_no_dataloaders_loaded(caplog):
    if False:
        while True:
            i = 10
    'Test that an info message is generated when dataloaders are loaded explicitly if they are not already\n    configured.'
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    trainer.fit_loop.setup_data()
    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64
    message = 'to estimate number of stepping batches'
    assert message not in caplog.text
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64
    assert message in caplog.text

def test_num_stepping_batches_iterable_dataset():
    if False:
        for i in range(10):
            print('nop')
    'Test the stepping batches with iterable dataset configured with max steps.'
    max_steps = 1000
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    train_dl = DataLoader(RandomIterableDataset(size=7, count=int(10000000000.0)))
    trainer._data_connector.attach_data(model, train_dataloaders=train_dl)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == max_steps

def test_num_stepping_batches_infinite_training():
    if False:
        while True:
            i = 10
    'Test that stepping batches is "inf" when `Trainer` is configured for infinite training.'
    trainer = Trainer(max_steps=-1, max_epochs=-1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == float('inf')

@pytest.mark.parametrize('max_steps', [2, 100])
def test_num_stepping_batches_with_max_steps(max_steps):
    if False:
        for i in range(10):
            print('nop')
    'Test stepping batches with `max_steps`.'
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    trainer.fit(model)
    assert trainer.estimated_stepping_batches == max_steps

@pytest.mark.parametrize(('accumulate_grad_batches', 'expected_steps'), [(2, 32), (3, 22)])
def test_num_stepping_batches_accumulate_gradients(accumulate_grad_batches, expected_steps):
    if False:
        print('Hello World!')
    'Test the total stepping batches when accumulation grad batches is configured.'
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=accumulate_grad_batches)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == expected_steps

@RunIf(mps=False)
@pytest.mark.parametrize(('trainer_kwargs', 'estimated_steps'), [({'strategy': 'ddp', 'num_nodes': 1}, 10), ({'strategy': 'ddp', 'num_nodes': 2}, 5), ({'strategy': 'ddp', 'num_nodes': 3}, 4), ({'strategy': 'ddp', 'num_nodes': 4}, 3)])
def test_num_stepping_batches_gpu(trainer_kwargs, estimated_steps, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Test stepping batches with GPU strategies.'
    num_devices_per_node = 7
    mock_cuda_count(monkeypatch, num_devices_per_node)
    trainer = Trainer(max_epochs=1, devices=num_devices_per_node, accelerator='gpu', **trainer_kwargs)
    trainer.strategy.parallel_devices = [torch.device('cpu', index=i) for i in range(num_devices_per_node)]
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == estimated_steps

@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_num_stepping_batches_with_tpu_single():
    if False:
        i = 10
        return i + 15
    'Test stepping batches with the single-core TPU strategy.'
    trainer = Trainer(accelerator='tpu', devices=1, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    assert isinstance(trainer.strategy, SingleDeviceXLAStrategy)
    trainer.strategy.connect(model)
    expected = len(model.train_dataloader())
    assert trainer.estimated_stepping_batches == expected

class MultiprocessModel(BoringModel):

    def on_train_start(self):
        if False:
            return 10
        assert self.trainer.estimated_stepping_batches == len(self.train_dataloader()) // self.trainer.world_size

@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_num_stepping_batches_with_tpu_multi():
    if False:
        for i in range(10):
            print('nop')
    'Test stepping batches with the TPU strategy across multiple devices.'
    trainer = Trainer(accelerator='tpu', devices='auto', max_epochs=1)
    model = MultiprocessModel()
    trainer.fit(model)