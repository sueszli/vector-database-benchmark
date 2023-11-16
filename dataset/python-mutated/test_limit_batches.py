import logging
import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.states import TrainerFn

def test_num_dataloader_batches(tmpdir):
    if False:
        print('Hello World!')
    'Tests that the correct number of batches are allocated.'
    model = BoringModel()
    trainer = Trainer(limit_val_batches=100, limit_train_batches=100, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 64
    assert trainer.num_training_batches == 64
    model = BoringModel()
    trainer = Trainer(limit_val_batches=7, limit_train_batches=7, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 7
    assert trainer.num_training_batches == 7

@pytest.mark.parametrize('mode', ['val', 'test', 'predict'])
@pytest.mark.parametrize('limit_batches', [0.1, 10])
def test_eval_limit_batches(mode, limit_batches):
    if False:
        while True:
            i = 10
    limit_eval_batches = f'limit_{mode}_batches'
    dl_hook = f'{mode}_dataloader'
    model = BoringModel()
    eval_loader = getattr(model, dl_hook)()
    trainer = Trainer(**{limit_eval_batches: limit_batches})
    model.trainer = trainer
    trainer.strategy.connect(model)
    trainer._data_connector.attach_dataloaders(model)
    if mode == 'val':
        trainer.validate_loop.setup_data()
        trainer.state.fn = TrainerFn.VALIDATING
        loader_num_batches = trainer.num_val_batches
        dataloaders = trainer.val_dataloaders
    elif mode == 'test':
        trainer.test_loop.setup_data()
        loader_num_batches = trainer.num_test_batches
        dataloaders = trainer.test_dataloaders
    elif mode == 'predict':
        trainer.predict_loop.setup_data()
        loader_num_batches = trainer.num_predict_batches
        dataloaders = trainer.predict_dataloaders
    expected_batches = int(limit_batches * len(eval_loader)) if isinstance(limit_batches, float) else limit_batches
    assert loader_num_batches[0] == expected_batches
    assert len(dataloaders) == len(eval_loader)

@pytest.mark.parametrize('argument', ['limit_train_batches', 'limit_val_batches', 'limit_test_batches', 'limit_predict_batches', 'overfit_batches'])
@pytest.mark.parametrize('value', [1, 1.0])
def test_limit_batches_info_message(caplog, argument, value):
    if False:
        return 10
    with caplog.at_level(logging.INFO):
        Trainer(**{argument: value})
    assert f'`Trainer({argument}={value})` was configured' in caplog.text
    message = f"configured so {('1' if isinstance(value, int) else '100%')}"
    assert message in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        Trainer()
    assert message not in caplog.text