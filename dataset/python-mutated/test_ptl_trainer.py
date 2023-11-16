import numpy as np
import pytest
from darts.logging import get_logger
from darts.tests.conftest import tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries
logger = get_logger(__name__)
try:
    import pytorch_lightning as pl
    from darts.models.forecasting.rnn_model import RNNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. RNN tests will be skipped.')
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:

    class TestTorchForecastingModel:
        trainer_params = {'max_epochs': 1, 'logger': False, 'enable_checkpointing': False}
        series = linear_timeseries(length=100).astype(np.float32)
        pl_200_or_above = int(pl.__version__.split('.')[0]) >= 2
        precisions = {32: '32' if not pl_200_or_above else '32-true', 64: '64' if not pl_200_or_above else '64-true'}

        def test_prediction_loaded_custom_trainer(self, tmpdir_module):
            if False:
                for i in range(10):
                    print('nop')
            'validate manual save with automatic save files by comparing output between the two'
            auto_name = 'test_save_automatic'
            model = RNNModel(12, 'RNN', 10, 10, model_name=auto_name, work_dir=tmpdir_module, save_checkpoints=True, random_state=42, **tfm_kwargs)
            trainer = pl.Trainer(max_epochs=1, enable_checkpointing=True, logger=False, callbacks=model.trainer_params['callbacks'], **tfm_kwargs['pl_trainer_kwargs'])
            model.fit(self.series, trainer=trainer)
            model_loaded = RNNModel.load_from_checkpoint(model_name=auto_name, work_dir=tmpdir_module, best=False, map_location='cpu')
            assert model.predict(n=4) == model_loaded.predict(n=4)

        def test_prediction_custom_trainer(self):
            if False:
                i = 10
                return i + 15
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, **tfm_kwargs)
            model2 = RNNModel(12, 'RNN', 10, 10, random_state=42, **tfm_kwargs)
            trainer = pl.Trainer(**self.trainer_params, precision=self.precisions[32], **tfm_kwargs['pl_trainer_kwargs'])
            model.fit(self.series, trainer=trainer)
            model2.fit(self.series, epochs=1)
            assert model.predict(n=4) == model2.predict(n=4)

        def test_custom_trainer_setup(self):
            if False:
                i = 10
                return i + 15
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, **tfm_kwargs)
            trainer = pl.Trainer(**self.trainer_params, precision=self.precisions[64], **tfm_kwargs['pl_trainer_kwargs'])
            with pytest.raises(ValueError):
                model.fit(self.series, trainer=trainer)
            trainer = pl.Trainer(**self.trainer_params, precision=self.precisions[32], **tfm_kwargs['pl_trainer_kwargs'])
            model.fit(self.series, trainer=trainer)
            assert trainer.max_epochs == model.epochs_trained

        def test_builtin_extended_trainer(self):
            if False:
                i = 10
                return i + 15
            with pytest.raises(TypeError):
                invalid_trainer_kwarg = {'precisionn': self.precisions[32], **tfm_kwargs['pl_trainer_kwargs']}
                model = RNNModel(12, 'RNN', 10, 10, random_state=42, pl_trainer_kwargs=invalid_trainer_kwarg)
                model.fit(self.series, epochs=1)
            with pytest.raises(ValueError):
                invalid_trainer_kwarg = {'precision': '16-mixed', **tfm_kwargs['pl_trainer_kwargs']}
                model = RNNModel(12, 'RNN', 10, 10, random_state=42, pl_trainer_kwargs=invalid_trainer_kwarg)
                model.fit(self.series.astype(np.float16), epochs=1)
            with pytest.raises(ValueError):
                invalid_trainer_kwarg = {'precision': self.precisions[64], **tfm_kwargs['pl_trainer_kwargs']}
                model = RNNModel(12, 'RNN', 10, 10, random_state=42, pl_trainer_kwargs=invalid_trainer_kwarg)
                model.fit(self.series.astype(np.float32), epochs=1)
            for precision in [64, 32]:
                valid_trainer_kwargs = {'precision': self.precisions[precision], **tfm_kwargs['pl_trainer_kwargs']}
                model = RNNModel(12, 'RNN', 10, 10, random_state=42, pl_trainer_kwargs=valid_trainer_kwargs)
                ts_dtype = getattr(np, f'float{precision}')
                model.fit(self.series.astype(ts_dtype), epochs=1)
                preds = model.predict(n=3)
                assert model.trainer.precision == self.precisions[precision]
                assert preds.dtype == ts_dtype

        def test_custom_callback(self, tmpdir_module):
            if False:
                while True:
                    i = 10

            class CounterCallback(pl.callbacks.Callback):

                def __init__(self, count_default):
                    if False:
                        i = 10
                        return i + 15
                    self.counter = count_default

                def on_train_epoch_end(self, *args, **kwargs):
                    if False:
                        print('Hello World!')
                    self.counter += 1
            my_counter_0 = CounterCallback(count_default=0)
            my_counter_2 = CounterCallback(count_default=2)
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, pl_trainer_kwargs={'callbacks': [my_counter_0, my_counter_2], **tfm_kwargs['pl_trainer_kwargs']})
            assert len(model.trainer_params['callbacks']) == 2
            model.fit(self.series, epochs=2, verbose=True)
            assert len(model.trainer_params['callbacks']) == 2
            assert my_counter_0.counter == model.epochs_trained
            assert my_counter_2.counter == model.epochs_trained + 2
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, work_dir=tmpdir_module, save_checkpoints=True, pl_trainer_kwargs={'callbacks': [CounterCallback(0), CounterCallback(2)], **tfm_kwargs['pl_trainer_kwargs']})
            assert len(model.trainer_params['callbacks']) == 3
            assert isinstance(model.trainer_params['callbacks'][0], pl.callbacks.ModelCheckpoint)
            for i in range(1, 3):
                assert isinstance(model.trainer_params['callbacks'][i], CounterCallback)

        def test_early_stopping(self):
            if False:
                print('Hello World!')
            my_stopper = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', stopping_threshold=1000000000.0)
            model = RNNModel(12, 'RNN', 10, 10, nr_epochs_val_period=1, random_state=42, pl_trainer_kwargs={'callbacks': [my_stopper], **tfm_kwargs['pl_trainer_kwargs']})
            model.fit(self.series, val_series=self.series, epochs=100, verbose=True)
            assert model.epochs_trained == 1
            my_stopper = pl.callbacks.early_stopping.EarlyStopping(monitor='invalid_variable', stopping_threshold=1000000000.0)
            model = RNNModel(12, 'RNN', 10, 10, nr_epochs_val_period=1, random_state=42, pl_trainer_kwargs={'callbacks': [my_stopper], **tfm_kwargs['pl_trainer_kwargs']})
            with pytest.raises(RuntimeError):
                model.fit(self.series, val_series=self.series, epochs=100, verbose=True)