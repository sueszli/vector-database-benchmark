from unittest.mock import ANY, Mock, call
import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

class DummyPredictionWriter(BasePredictionWriter):

    def write_on_batch_end(self, *_, **__):
        if False:
            print('Hello World!')
        pass

    def write_on_epoch_end(self, *_, **__):
        if False:
            while True:
                i = 10
        pass

def test_prediction_writer_invalid_write_interval():
    if False:
        i = 10
        return i + 15
    'Test that configuring an unknown interval name raises an error.'
    with pytest.raises(MisconfigurationException, match="`write_interval` should be one of \\['batch"):
        DummyPredictionWriter('something')

def test_prediction_writer_hook_call_intervals():
    if False:
        i = 10
        return i + 15
    'Test that the `write_on_batch_end` and `write_on_epoch_end` hooks get invoked based on the defined interval.'
    DummyPredictionWriter.write_on_batch_end = Mock()
    DummyPredictionWriter.write_on_epoch_end = Mock()
    dataloader = DataLoader(RandomDataset(32, 64))
    model = BoringModel()
    cb = DummyPredictionWriter('batch_and_epoch')
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    results = trainer.predict(model, dataloaders=dataloader)
    assert len(results) == 4
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1
    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()
    cb = DummyPredictionWriter('batch_and_epoch')
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1
    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()
    cb = DummyPredictionWriter('batch')
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 0
    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()
    cb = DummyPredictionWriter('epoch')
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 0
    assert cb.write_on_epoch_end.call_count == 1

@pytest.mark.parametrize('num_workers', [0, 2])
def test_prediction_writer_batch_indices(num_workers):
    if False:
        print('Hello World!')
    DummyPredictionWriter.write_on_batch_end = Mock()
    DummyPredictionWriter.write_on_epoch_end = Mock()
    dataloader = DataLoader(RandomDataset(32, 64), batch_size=4, num_workers=num_workers)
    model = BoringModel()
    writer = DummyPredictionWriter('batch_and_epoch')
    trainer = Trainer(limit_predict_batches=4, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader)
    writer.write_on_batch_end.assert_has_calls([call(trainer, model, ANY, [0, 1, 2, 3], ANY, 0, 0), call(trainer, model, ANY, [4, 5, 6, 7], ANY, 1, 0), call(trainer, model, ANY, [8, 9, 10, 11], ANY, 2, 0), call(trainer, model, ANY, [12, 13, 14, 15], ANY, 3, 0)])
    writer.write_on_epoch_end.assert_has_calls([call(trainer, model, ANY, [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]])])

def test_batch_level_batch_indices():
    if False:
        print('Hello World!')
    'Test that batch_indices are returned when `return_predictions=False`.'
    DummyPredictionWriter.write_on_batch_end = Mock()

    class CustomBoringModel(BoringModel):

        def on_predict_epoch_end(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            assert self.trainer.predict_loop.epoch_batch_indices == [[]]
    writer = DummyPredictionWriter('batch')
    model = CustomBoringModel()
    dataloader = DataLoader(RandomDataset(32, 64), batch_size=4)
    trainer = Trainer(limit_predict_batches=4, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    writer.write_on_batch_end.assert_has_calls([call(trainer, model, ANY, [0, 1, 2, 3], ANY, 0, 0), call(trainer, model, ANY, [4, 5, 6, 7], ANY, 1, 0), call(trainer, model, ANY, [8, 9, 10, 11], ANY, 2, 0), call(trainer, model, ANY, [12, 13, 14, 15], ANY, 3, 0)])