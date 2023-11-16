import os
from typing import Any, Dict, Optional
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.encoders import SequentialEncoder
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.logging import get_logger
from darts.metrics import mape
from darts.tests.conftest import tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries
logger = get_logger(__name__)
try:
    import torch
    from pytorch_lightning.loggers.logger import DummyLogger
    from pytorch_lightning.tuner.lr_finder import _LRFinder
    from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MetricCollection
    from darts.models import BlockRNNModel, DLinearModel, NBEATSModel, NHiTSModel, NLinearModel, RNNModel, TCNModel, TFTModel, TiDEModel, TransformerModel
    from darts.models.components.layer_norm_variants import RINorm
    from darts.utils.likelihood_models import GaussianLikelihood, LaplaceLikelihood, Likelihood
    kwargs = {'input_chunk_length': 10, 'output_chunk_length': 1, 'n_epochs': 1, 'pl_trainer_kwargs': {'fast_dev_run': True, **tfm_kwargs['pl_trainer_kwargs']}}
    models = [(BlockRNNModel, kwargs), (DLinearModel, kwargs), (NBEATSModel, kwargs), (NHiTSModel, kwargs), (NLinearModel, kwargs), (RNNModel, {'training_length': 2, **kwargs}), (TCNModel, kwargs), (TFTModel, {'add_relative_index': 2, **kwargs}), (TiDEModel, kwargs), (TransformerModel, kwargs)]
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. RNN tests will be skipped.')
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:

    class TestTorchForecastingModel:
        times = pd.date_range('20130101', '20130410')
        pd_series = pd.Series(range(100), index=times)
        series = TimeSeries.from_series(pd_series)
        df = pd.DataFrame({'var1': range(100), 'var2': range(100)}, index=times)
        multivariate_series = TimeSeries.from_dataframe(df)

        def test_save_model_parameters(self):
            if False:
                i = 10
                return i + 15
            model = RNNModel(12, 'RNN', 10, 10, **tfm_kwargs)
            assert model._model_params, model.untrained_model()._model_params

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.save')
        def test_suppress_automatic_save(self, patch_save_model, tmpdir_fn):
            if False:
                return 10
            model_name = 'test_model'
            model1 = RNNModel(12, 'RNN', 10, 10, model_name=model_name, work_dir=tmpdir_fn, save_checkpoints=False, **tfm_kwargs)
            model2 = RNNModel(12, 'RNN', 10, 10, model_name=model_name, work_dir=tmpdir_fn, force_reset=True, save_checkpoints=False, **tfm_kwargs)
            model1.fit(self.series, epochs=1)
            model2.fit(self.series, epochs=1)
            model1.predict(n=1)
            model2.predict(n=2)
            patch_save_model.assert_not_called()
            model1.save(path=os.path.join(tmpdir_fn, model_name))
            patch_save_model.assert_called()

        def test_manual_save_and_load(self, tmpdir_fn):
            if False:
                while True:
                    i = 10
            'validate manual save with automatic save files by comparing output between the two'
            model_dir = os.path.join(tmpdir_fn)
            manual_name = 'test_save_manual'
            auto_name = 'test_save_automatic'
            model_manual_save = RNNModel(12, 'RNN', 10, 10, model_name=manual_name, work_dir=tmpdir_fn, save_checkpoints=False, random_state=42, **tfm_kwargs)
            model_auto_save = RNNModel(12, 'RNN', 10, 10, model_name=auto_name, work_dir=tmpdir_fn, save_checkpoints=True, random_state=42, **tfm_kwargs)
            no_training_ckpt = 'no_training.pth.tar'
            no_training_ckpt_path = os.path.join(model_dir, no_training_ckpt)
            model_manual_save.save(no_training_ckpt_path)
            assert os.path.exists(no_training_ckpt_path)
            assert not os.path.exists(no_training_ckpt_path + '.ckpt')
            with pytest.raises(ValueError) as err:
                no_train_model = RNNModel.load(no_training_ckpt_path)
                no_train_model.predict(n=4)
            assert str(err.value) == 'Input `series` must be provided. This is the result either from fitting on multiple series, or from not having fit the model yet.'
            model_manual_save.fit(self.series, epochs=1)
            model_auto_save.fit(self.series, epochs=1)
            assert not os.path.exists(os.path.join(model_dir, manual_name, 'checkpoints'))
            assert os.path.exists(os.path.join(model_dir, auto_name, 'checkpoints'))
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = 'checkpoint_0.pth.tar'
            model_path_manual = os.path.join(checkpoint_path_manual, checkpoint_file_name)
            checkpoint_file_name_cpkt = 'checkpoint_0.pth.tar.ckpt'
            model_path_manual_ckpt = os.path.join(checkpoint_path_manual, checkpoint_file_name_cpkt)
            model_manual_save.save(model_path_manual)
            assert os.path.exists(model_path_manual)
            assert os.path.exists(model_path_manual_ckpt)
            model_manual_save = RNNModel.load(model_path_manual, map_location='cpu')
            model_manual_save.to_cpu()
            assert model_manual_save.predict(n=4) == model_auto_save.predict(n=4)
            model_auto_save1 = RNNModel.load_from_checkpoint(model_name=auto_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            model_auto_save1.to_cpu()
            assert model_manual_save.predict(n=4) == model_auto_save1.predict(n=4)
            checkpoint_file_name_2 = 'checkpoint_1.pth.tar'
            checkpoint_file_name_cpkt_2 = checkpoint_file_name_2 + '.ckpt'
            model_path_manual_2 = os.path.join(checkpoint_path_manual, checkpoint_file_name_2)
            model_path_manual_ckpt_2 = os.path.join(checkpoint_path_manual, checkpoint_file_name_cpkt_2)
            model_auto_save2 = RNNModel.load_from_checkpoint(model_name=auto_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            model_auto_save2.save(model_path_manual_2)
            assert os.path.exists(model_path_manual_ckpt_2)
            model_chained_load_save = RNNModel.load(model_path_manual_2, map_location='cpu')
            assert model_chained_load_save.predict(n=4) == model_manual_save.predict(n=4)

        def test_valid_save_and_load_weights_with_different_params(self, tmpdir_fn):
            if False:
                while True:
                    i = 10
            '\n            Verify that save/load does not break encoders.\n\n            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used\n            for all but one test.\n            Note: Using DLinear since it supports both past and future covariates\n            '

            def create_model(**kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return DLinearModel(input_chunk_length=4, output_chunk_length=1, **kwargs, **tfm_kwargs)
            model_dir = os.path.join(tmpdir_fn)
            manual_name = 'save_manual'
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = 'checkpoint_0.pth.tar'
            model_path_manual = os.path.join(checkpoint_path_manual, checkpoint_file_name)
            model = create_model()
            model.fit(self.series, epochs=1)
            model.save(model_path_manual)
            kwargs_valid = [{'optimizer_cls': torch.optim.SGD}, {'optimizer_kwargs': {'lr': 0.1}}]
            for kwargs_ in kwargs_valid:
                model_new = create_model(**kwargs_)
                model_new.load_weights(model_path_manual)

        def test_save_and_load_weights_w_encoders(self, tmpdir_fn):
            if False:
                print('Hello World!')
            '\n            Verify that save/load does not break encoders.\n\n            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used\n            for all but one test.\n            Note: Using DLinear since it supports both past and future covariates\n            '
            model_dir = os.path.join(tmpdir_fn)
            manual_name = 'save_manual'
            auto_name = 'save_auto'
            auto_name_other = 'save_auto_other'
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = 'checkpoint_0.pth.tar'
            model_path_manual = os.path.join(checkpoint_path_manual, checkpoint_file_name)
            encoders_past = {'datetime_attribute': {'past': ['day']}, 'transformer': Scaler()}
            encoders_other_past = {'datetime_attribute': {'past': ['hour']}, 'transformer': Scaler()}
            encoders_past_noscaler = {'datetime_attribute': {'past': ['day']}}
            encoders_past_other_transformer = {'datetime_attribute': {'past': ['day']}, 'transformer': BoxCox()}
            encoders_2_past = {'datetime_attribute': {'past': ['hour', 'day']}, 'transformer': Scaler()}
            encoders_past_n_future = {'datetime_attribute': {'past': ['day'], 'future': ['dayofweek']}, 'transformer': Scaler()}
            model_auto_save = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name=auto_name, save_checkpoints=True, add_encoders=encoders_past)
            model_auto_save.fit(self.series, epochs=1)
            model_manual_save = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name=manual_name, save_checkpoints=False, add_encoders=encoders_past)
            model_manual_save.fit(self.series, epochs=1)
            model_manual_save.save(model_path_manual)
            model_auto_save_other = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name=auto_name_other, save_checkpoints=True, add_encoders=encoders_other_past)
            model_auto_save_other.fit(self.series, epochs=1)
            assert model_auto_save.predict(n=4) != model_auto_save_other.predict(n=4)
            model_no_enc = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='no_encoder', add_encoders=None)
            with pytest.raises(ValueError):
                model_no_enc.load_weights_from_checkpoint(auto_name, work_dir=tmpdir_fn, best=False, load_encoders=False, map_location='cpu')
            model_no_enc.load_weights_from_checkpoint(auto_name, work_dir=tmpdir_fn, best=False, load_encoders=True, map_location='cpu')
            self.helper_equality_encoders(model_auto_save.add_encoders, model_no_enc.add_encoders)
            self.helper_equality_encoders_transfo(model_auto_save.add_encoders, model_no_enc.add_encoders)
            assert model_auto_save.predict(n=4) == model_no_enc.predict(n=4, series=self.series)
            model_same_enc_noload = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_encoder_noload', add_encoders=encoders_past)
            model_same_enc_noload.load_weights(model_path_manual, load_encoders=False, map_location='cpu')
            with pytest.raises(ValueError):
                model_same_enc_noload.predict(n=4, series=self.series)
            model_same_enc_load = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_encoder_load', add_encoders=encoders_past)
            model_same_enc_load.load_weights(model_path_manual, load_encoders=True, map_location='cpu')
            assert model_manual_save.predict(n=4) == model_same_enc_load.predict(n=4, series=self.series)
            model_other_enc_load = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='other_encoder_load', add_encoders=encoders_other_past)
            with pytest.raises(ValueError):
                model_other_enc_load.load_weights(model_path_manual, load_encoders=True, map_location='cpu')
            model_other_enc_noload = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='other_encoder_noload', add_encoders=encoders_other_past)
            model_other_enc_noload.load_weights(model_path_manual, load_encoders=False, map_location='cpu')
            self.helper_equality_encoders(model_other_enc_noload.add_encoders, encoders_other_past)
            self.helper_equality_encoders_transfo(model_other_enc_noload.add_encoders, encoders_other_past)
            assert isinstance(model_other_enc_noload.encoders, SequentialEncoder)
            with pytest.raises(ValueError):
                model_other_enc_noload.predict(n=4, series=self.series)
            model_other_enc_noload.fit(self.series, epochs=1)
            model_other_enc_noload.predict(n=4, series=self.series)
            model_new_enc_noscaler_noload = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_encoder_noscaler', add_encoders=encoders_past_noscaler)
            model_new_enc_noscaler_noload.load_weights(model_path_manual, load_encoders=False, map_location='cpu')
            self.helper_equality_encoders(model_new_enc_noscaler_noload.add_encoders, encoders_past_noscaler)
            self.helper_equality_encoders_transfo(model_new_enc_noscaler_noload.add_encoders, encoders_past_noscaler)
            model_new_enc_noscaler_noload.predict(n=4, series=self.series)
            model_new_enc_other_transformer = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_encoder_other_transform', add_encoders=encoders_past_other_transformer)
            with pytest.raises(ValueError):
                model_new_enc_other_transformer.load_weights(model_path_manual, load_encoders=True, map_location='cpu')
            model_new_enc_other_transformer.load_weights(model_path_manual, load_encoders=False, map_location='cpu')
            with pytest.raises(ValueError):
                model_new_enc_other_transformer.predict(n=4, series=self.series)
            model_new_enc_other_transformer.fit(self.series, epochs=1)
            model_new_enc_other_transformer.predict(n=4, series=self.series)
            model_new_enc_2_past = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='encoder_2_components_past', add_encoders=encoders_2_past)
            with pytest.raises(ValueError):
                model_new_enc_2_past.load_weights(model_path_manual, load_encoders=True, map_location='cpu')
            with pytest.raises(ValueError):
                model_new_enc_2_past.load_weights(model_path_manual, load_encoders=False, map_location='cpu')
            model_new_enc_past_n_future = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='encoder_past_n_future', add_encoders=encoders_past_n_future)
            with pytest.raises(ValueError):
                model_new_enc_past_n_future.load_weights(model_path_manual, load_encoders=True, map_location='cpu')
            with pytest.raises(ValueError):
                model_new_enc_past_n_future.load_weights(model_path_manual, load_encoders=False, map_location='cpu')

        def test_save_and_load_weights_w_likelihood(self, tmpdir_fn):
            if False:
                i = 10
                return i + 15
            '\n            Verify that save/load does not break likelihood.\n\n            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used\n            for all but one test.\n            Note: Using DLinear since it supports both past and future covariates\n            '
            model_dir = os.path.join(tmpdir_fn)
            manual_name = 'save_manual'
            auto_name = 'save_auto'
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = 'checkpoint_0.pth.tar'
            model_path_manual = os.path.join(checkpoint_path_manual, checkpoint_file_name)
            model_auto_save = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name=auto_name, save_checkpoints=True, likelihood=GaussianLikelihood(prior_mu=0.5))
            model_auto_save.fit(self.series, epochs=1)
            pred_auto = model_auto_save.predict(n=4, series=self.series)
            model_manual_save = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name=manual_name, save_checkpoints=False, likelihood=GaussianLikelihood(prior_mu=0.5))
            model_manual_save.fit(self.series, epochs=1)
            model_manual_save.save(model_path_manual)
            pred_manual = model_manual_save.predict(n=4, series=self.series)
            assert np.array_equal(pred_auto.values(), pred_manual.values())
            model_same_likelihood = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_likelihood', likelihood=GaussianLikelihood(prior_mu=0.5))
            model_same_likelihood.load_weights(model_path_manual, map_location='cpu')
            model_same_likelihood.predict(n=4, series=self.series)
            model_manual_same_likelihood = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_likelihood', likelihood=GaussianLikelihood(prior_mu=0.5))
            model_manual_same_likelihood.load_weights(model_path_manual, map_location='cpu')
            preds_manual_from_weights = model_manual_same_likelihood.predict(n=4, series=self.series)
            model_auto_same_likelihood = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_likelihood', likelihood=GaussianLikelihood(prior_mu=0.5))
            model_auto_same_likelihood.load_weights_from_checkpoint(auto_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            preds_auto_from_weights = model_auto_same_likelihood.predict(n=4, series=self.series)
            assert preds_manual_from_weights == preds_auto_from_weights
            model_no_likelihood = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='no_likelihood', likelihood=None)
            with pytest.raises(ValueError) as error_msg:
                model_no_likelihood.load_weights_from_checkpoint(auto_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nincorrect')
            model_no_likelihood_bis = DLinearModel(input_chunk_length=4, output_chunk_length=1, model_name='no_likelihood_bis', add_encoders=None, work_dir=tmpdir_fn, save_checkpoints=False, random_state=42, force_reset=True, n_epochs=1, **tfm_kwargs)
            with pytest.raises(ValueError) as error_msg:
                model_no_likelihood_bis.load_weights_from_checkpoint(auto_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nmissing')
            model_other_likelihood = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='other_likelihood', likelihood=LaplaceLikelihood())
            with pytest.raises(ValueError) as error_msg:
                model_other_likelihood.load_weights(model_path_manual, map_location='cpu')
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nincorrect')
            model_same_likelihood_other_prior = self.helper_create_DLinearModel(work_dir=tmpdir_fn, model_name='same_likelihood_other_prior', likelihood=GaussianLikelihood())
            with pytest.raises(ValueError) as error_msg:
                model_same_likelihood_other_prior.load_weights(model_path_manual, map_location='cpu')
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nincorrect')

        def test_load_weights_params_check(self, tmpdir_fn):
            if False:
                while True:
                    i = 10
            '\n            Verify that the method comparing the parameters between the saved model and the loading model\n            behave as expected, used to return meaningful error message instead of the torch.load ones.\n            '
            model_name = 'params_check'
            ckpt_path = os.path.join(tmpdir_fn, f'{model_name}.pt')
            model = DLinearModel(input_chunk_length=4, output_chunk_length=1, n_epochs=1)
            model.fit(self.series[:10])
            model.save(ckpt_path)
            loading_model = DLinearModel(input_chunk_length=4, output_chunk_length=1)
            loading_model.load_weights(ckpt_path)
            loading_model = DLinearModel(input_chunk_length=4, output_chunk_length=1, optimizer_cls=torch.optim.AdamW)
            loading_model.load_weights(ckpt_path)
            loading_model = DLinearModel(input_chunk_length=4, output_chunk_length=1, pl_trainer_kwargs={'enable_model_summary': False})
            loading_model.load_weights(ckpt_path)
            loading_model = DLinearModel(input_chunk_length=4 + 1, output_chunk_length=1)
            with pytest.raises(ValueError) as error_msg:
                loading_model.load_weights(ckpt_path)
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nincorrect')
            loading_model = DLinearModel(input_chunk_length=4, output_chunk_length=1, kernel_size=10)
            with pytest.raises(ValueError) as error_msg:
                loading_model.load_weights(ckpt_path)
            assert str(error_msg.value).startswith('The values of the hyper-parameters in the model and loaded checkpoint should be identical.\nincorrect')

        def test_create_instance_new_model_no_name_set(self, tmpdir_fn):
            if False:
                print('Hello World!')
            RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, **tfm_kwargs)

        def test_create_instance_existing_model_with_name_no_fit(self, tmpdir_fn):
            if False:
                print('Hello World!')
            model_name = 'test_model'
            RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, model_name=model_name, **tfm_kwargs)

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model')
        def test_create_instance_existing_model_with_name_force(self, patch_reset_model, tmpdir_fn):
            if False:
                while True:
                    i = 10
            model_name = 'test_model'
            RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, model_name=model_name, **tfm_kwargs)
            RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, model_name=model_name, force_reset=True, **tfm_kwargs)
            patch_reset_model.assert_not_called()

        @patch('darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model')
        def test_create_instance_existing_model_with_name_force_fit_with_reset(self, patch_reset_model, tmpdir_fn):
            if False:
                i = 10
                return i + 15
            model_name = 'test_model'
            model1 = RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, model_name=model_name, save_checkpoints=True, **tfm_kwargs)
            model1.fit(self.series, epochs=1)
            RNNModel(12, 'RNN', 10, 10, work_dir=tmpdir_fn, model_name=model_name, save_checkpoints=True, force_reset=True, **tfm_kwargs)
            patch_reset_model.assert_called_once()

        def test_train_from_0_n_epochs_20_no_fit_epochs(self):
            if False:
                return 10
            model1 = RNNModel(12, 'RNN', 10, 10, n_epochs=20, **tfm_kwargs)
            model1.fit(self.series)
            assert 20 == model1.epochs_trained

        def test_train_from_20_n_epochs_40_no_fit_epochs(self):
            if False:
                i = 10
                return i + 15
            model1 = RNNModel(12, 'RNN', 10, 10, n_epochs=20, **tfm_kwargs)
            model1.fit(self.series)
            assert 20 == model1.epochs_trained
            model1.fit(self.series)
            assert 20 == model1.epochs_trained

        def test_train_from_10_n_epochs_20_no_fit_epochs(self):
            if False:
                return 10
            model1 = RNNModel(12, 'RNN', 10, 10, n_epochs=20, **tfm_kwargs)
            model1.fit(self.series, epochs=10)
            assert 10 == model1.epochs_trained
            model1.fit(self.series)
            assert 20 == model1.epochs_trained

        def test_train_from_10_n_epochs_20_fit_15_epochs(self):
            if False:
                while True:
                    i = 10
            model1 = RNNModel(12, 'RNN', 10, 10, n_epochs=20, **tfm_kwargs)
            model1.fit(self.series, epochs=10)
            assert 10 == model1.epochs_trained
            model1.fit(self.series, epochs=15)
            assert 15 == model1.epochs_trained

        def test_load_weights_from_checkpoint(self, tmpdir_fn):
            if False:
                for i in range(10):
                    print('nop')
            (ts_training, ts_test) = self.series.split_before(90)
            original_model_name = 'original'
            retrained_model_name = 'retrained'
            model = RNNModel(12, 'RNN', 5, 1, n_epochs=5, work_dir=tmpdir_fn, save_checkpoints=True, model_name=original_model_name, random_state=1, **tfm_kwargs)
            model.fit(ts_training)
            original_preds = model.predict(10)
            original_mape = mape(original_preds, ts_test)
            model_rt = RNNModel(12, 'RNN', 5, 1, n_epochs=5, work_dir=tmpdir_fn, model_name=retrained_model_name, random_state=1, **tfm_kwargs)
            model_rt.load_weights_from_checkpoint(model_name=original_model_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            loaded_preds = model_rt.predict(10, ts_training)
            assert original_preds == loaded_preds
            model_rt.fit(ts_training)
            retrained_preds = model_rt.predict(10)
            retrained_mape = mape(retrained_preds, ts_test)
            assert retrained_mape < original_mape, f'Retrained model has a greater error (mape) than the original model, respectively {retrained_mape} and {original_mape}'
            with pytest.raises(ValueError):
                model_rt = RNNModel(12, 'RNN', 10, 5)
                model_rt.load_weights_from_checkpoint(model_name=original_model_name, work_dir=tmpdir_fn, best=False, map_location='cpu')
            with pytest.raises(ValueError):
                model_rt = RNNModel(12, 'RNN', 5, 5, **tfm_kwargs)
                model_rt.load_weights_from_checkpoint(model_name=original_model_name, work_dir=tmpdir_fn, best=False, weights_only=True, map_location='cpu')

        def test_load_weights(self, tmpdir_fn):
            if False:
                while True:
                    i = 10
            (ts_training, ts_test) = self.series.split_before(90)
            original_model_name = 'original'
            retrained_model_name = 'retrained'
            model = RNNModel(12, 'RNN', 5, 1, n_epochs=5, work_dir=tmpdir_fn, save_checkpoints=False, model_name=original_model_name, random_state=1, **tfm_kwargs)
            model.fit(ts_training)
            path_manual_save = os.path.join(tmpdir_fn, 'RNN_manual_save.pt')
            model.save(path_manual_save)
            original_preds = model.predict(10)
            original_mape = mape(original_preds, ts_test)
            model_rt = RNNModel(12, 'RNN', 5, 1, n_epochs=5, work_dir=tmpdir_fn, model_name=retrained_model_name, random_state=1, **tfm_kwargs)
            model_rt.load_weights(path=path_manual_save, map_location='cpu')
            loaded_preds = model_rt.predict(10, ts_training)
            assert original_preds == loaded_preds
            model_rt.fit(ts_training)
            retrained_preds = model_rt.predict(10)
            retrained_mape = mape(retrained_preds, ts_test)
            assert retrained_mape < original_mape, f'Retrained model has a greater mape error than the original model, respectively {retrained_mape} and {original_mape}'

        def test_load_weights_with_float32_dtype(self, tmpdir_fn):
            if False:
                return 10
            ts_float32 = self.series.astype('float32')
            model_name = 'test_model'
            ckpt_path = os.path.join(tmpdir_fn, f'{model_name}.pt')
            model = DLinearModel(input_chunk_length=4, output_chunk_length=1, n_epochs=1)
            model.fit(ts_float32)
            model.save(ckpt_path)
            assert model.model._dtype == torch.float32
            loading_model = DLinearModel(input_chunk_length=4, output_chunk_length=1)
            loading_model.load_weights(ckpt_path)
            loading_model.fit(ts_float32)
            assert loading_model.model._dtype == torch.float32

        def test_multi_steps_pipeline(self, tmpdir_fn):
            if False:
                print('Hello World!')
            (ts_training, ts_val) = self.series.split_before(75)
            pretrain_model_name = 'pre-train'
            retrained_model_name = 're-train'
            model = self.helper_create_RNNModel(pretrain_model_name, tmpdir_fn)
            model.fit(ts_training, val_series=ts_val)
            model = self.helper_create_RNNModel(retrained_model_name, tmpdir_fn)
            model.load_weights_from_checkpoint(model_name=pretrain_model_name, work_dir=tmpdir_fn, best=True, map_location='cpu')
            model.fit(ts_training, val_series=ts_val)
            model = model.load_from_checkpoint(model_name=retrained_model_name, work_dir=tmpdir_fn, best=True, map_location='cpu')
            model.predict(4, series=ts_training)

        def test_load_from_checkpoint_w_custom_loss(self, tmpdir_fn):
            if False:
                while True:
                    i = 10
            model_name = 'pretraining_custom_loss'
            model = RNNModel(12, 'RNN', 5, 1, n_epochs=1, work_dir=tmpdir_fn, model_name=model_name, save_checkpoints=True, force_reset=True, loss_fn=torch.nn.L1Loss(), **tfm_kwargs)
            model.fit(self.series)
            loaded_model = RNNModel.load_from_checkpoint(model_name, tmpdir_fn, best=False, map_location='cpu')
            assert isinstance(loaded_model.model.criterion, torch.nn.L1Loss)
            loaded_model.fit(self.series, epochs=2)
            assert isinstance(loaded_model.model.criterion, torch.nn.L1Loss)

        def test_load_from_checkpoint_w_metrics(self, tmpdir_fn):
            if False:
                i = 10
                return i + 15
            model_name = 'pretraining_metrics'
            pl_trainer_kwargs = dict({'logger': DummyLogger(), 'log_every_n_steps': 1}, **tfm_kwargs['pl_trainer_kwargs'])
            model = RNNModel(12, 'RNN', 5, 1, n_epochs=1, work_dir=tmpdir_fn, model_name=model_name, save_checkpoints=True, force_reset=True, torch_metrics=MeanAbsolutePercentageError(), pl_trainer_kwargs=pl_trainer_kwargs)
            model.fit(self.series)
            assert isinstance(model.model.train_metrics, MetricCollection)
            assert len(model.model.train_metrics) == 1
            loaded_model = RNNModel.load_from_checkpoint(model_name, tmpdir_fn, best=False, map_location='cpu')
            assert isinstance(loaded_model.model.train_metrics, MetricCollection)
            assert len(loaded_model.model.train_metrics) == 1

        def test_optimizers(self):
            if False:
                while True:
                    i = 10
            optimizers = [(torch.optim.Adam, {'lr': 0.001}), (torch.optim.SGD, {'lr': 0.001})]
            for (optim_cls, optim_kwargs) in optimizers:
                model = RNNModel(12, 'RNN', 10, 10, optimizer_cls=optim_cls, optimizer_kwargs=optim_kwargs, **tfm_kwargs)
                model.fit(self.series, epochs=1)

        def test_lr_schedulers(self):
            if False:
                for i in range(10):
                    print('nop')
            lr_schedulers = [(torch.optim.lr_scheduler.StepLR, {'step_size': 10}), (torch.optim.lr_scheduler.ReduceLROnPlateau, {'threshold': 0.001, 'monitor': 'train_loss'}), (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.09})]
            for (lr_scheduler_cls, lr_scheduler_kwargs) in lr_schedulers:
                model = RNNModel(12, 'RNN', 10, 10, lr_scheduler_cls=lr_scheduler_cls, lr_scheduler_kwargs=lr_scheduler_kwargs, **tfm_kwargs)
                model.fit(self.series, epochs=1)

        def test_wrong_model_creation_params(self):
            if False:
                for i in range(10):
                    print('nop')
            valid_kwarg = {'pl_trainer_kwargs': {}}
            invalid_kwarg = {'some_invalid_kwarg': None}
            _ = RNNModel(12, 'RNN', 10, 10, **valid_kwarg)
            with pytest.raises(ValueError):
                _ = RNNModel(12, 'RNN', 10, 10, **invalid_kwarg)

        def test_metrics(self):
            if False:
                i = 10
                return i + 15
            metric = MeanAbsolutePercentageError()
            metric_collection = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
            model_kwargs = {'logger': DummyLogger(), 'log_every_n_steps': 1, **tfm_kwargs['pl_trainer_kwargs']}
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, torch_metrics=metric, pl_trainer_kwargs=model_kwargs)
            model.fit(self.series)
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, torch_metrics=metric_collection, pl_trainer_kwargs=model_kwargs)
            model.fit(self.series)
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, torch_metrics=metric, pl_trainer_kwargs=model_kwargs)
            model.fit(self.multivariate_series)

        def test_metrics_w_likelihood(self):
            if False:
                return 10
            metric = MeanAbsolutePercentageError()
            metric_collection = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])
            model_kwargs = {'logger': DummyLogger(), 'log_every_n_steps': 1, **tfm_kwargs['pl_trainer_kwargs']}
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, likelihood=GaussianLikelihood(), torch_metrics=metric, pl_trainer_kwargs=model_kwargs)
            model.fit(self.series)
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, likelihood=GaussianLikelihood(), torch_metrics=metric_collection, pl_trainer_kwargs=model_kwargs)
            model.fit(self.series)
            model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, likelihood=GaussianLikelihood(), torch_metrics=metric_collection, pl_trainer_kwargs=model_kwargs)
            model.fit(self.multivariate_series)

        def test_invalid_metrics(self):
            if False:
                print('Hello World!')
            torch_metrics = ['invalid']
            with pytest.raises(AttributeError):
                model = RNNModel(12, 'RNN', 10, 10, n_epochs=1, torch_metrics=torch_metrics, **tfm_kwargs)
                model.fit(self.series)

        @pytest.mark.slow
        def test_lr_find(self):
            if False:
                while True:
                    i = 10
            (train_series, val_series) = (self.series[:-40], self.series[-40:])
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, **tfm_kwargs)
            res = model.lr_find(series=train_series, val_series=val_series, epochs=50)
            assert isinstance(res, _LRFinder)
            assert res.suggestion() is not None
            assert model.model is None
            assert not model._fit_called
            with pytest.raises(ValueError):
                model.predict(n=3, series=self.series)
            model = RNNModel(12, 'RNN', 10, 10, random_state=42, **tfm_kwargs)
            res2 = model.lr_find(series=train_series, val_series=val_series, epochs=50)
            assert res.suggestion() == res2.suggestion()
            lr_worst = res.results['lr'][np.argmax(res.results['loss'])]
            lr_suggested = res.suggestion()
            scores = {}
            for (lr, lr_name) in zip([lr_worst, lr_suggested], ['worst', 'suggested']):
                model = RNNModel(12, 'RNN', 10, 10, n_epochs=10, random_state=42, optimizer_cls=torch.optim.Adam, optimizer_kwargs={'lr': lr}, **tfm_kwargs)
                model.fit(train_series)
                scores[lr_name] = mape(val_series, model.predict(len(val_series), series=train_series))
            assert scores['worst'] > scores['suggested']

        def test_encoders(self, tmpdir_fn):
            if False:
                for i in range(10):
                    print('nop')
            series = linear_timeseries(length=10)
            pc = linear_timeseries(length=12)
            fc = linear_timeseries(length=13)
            ns = [1, 3]
            model = self.helper_create_DLinearModel(work_dir=tmpdir_fn, add_encoders={'datetime_attribute': {'past': ['hour'], 'future': ['month']}})
            model.fit(series)
            for n in ns:
                _ = model.predict(n=n)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)
            model = self.helper_create_DLinearModel(work_dir=tmpdir_fn, add_encoders={'datetime_attribute': {'past': ['hour'], 'future': ['month']}})
            for n in ns:
                model.fit(series, past_covariates=pc)
                _ = model.predict(n=n)
                _ = model.predict(n=n, past_covariates=pc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)
            model = self.helper_create_DLinearModel(work_dir=tmpdir_fn, add_encoders={'datetime_attribute': {'past': ['hour'], 'future': ['month']}})
            for n in ns:
                model.fit(series, future_covariates=fc)
                _ = model.predict(n=n)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc)
                _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)
            model = self.helper_create_DLinearModel(work_dir=tmpdir_fn, add_encoders={'datetime_attribute': {'past': ['hour'], 'future': ['month']}})
            for n in ns:
                model.fit(series, past_covariates=pc, future_covariates=fc)
                _ = model.predict(n=n)
                _ = model.predict(n=n, past_covariates=pc)
                _ = model.predict(n=n, future_covariates=fc)
                _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

        @pytest.mark.parametrize('model_config', models)
        def test_rin(self, model_config):
            if False:
                while True:
                    i = 10
            (model_cls, kwargs) = model_config
            model_no_rin = model_cls(use_reversible_instance_norm=False, **kwargs)
            model_rin = model_cls(use_reversible_instance_norm=True, **kwargs)
            model_no_rin.fit(self.series)
            assert not model_no_rin.model.use_reversible_instance_norm
            assert model_no_rin.model.rin is None
            model_rin.fit(self.series)
            if issubclass(model_cls, RNNModel):
                assert not model_rin.model.use_reversible_instance_norm
                assert model_rin.model.rin is None
                return
            else:
                assert model_rin.model.use_reversible_instance_norm
                assert isinstance(model_rin.model.rin, RINorm)
                assert model_rin.model.rin.input_dim == self.series.n_components
            model_rin_mv = model_rin.untrained_model()
            model_rin_mv.fit(self.multivariate_series)
            assert model_rin_mv.model.use_reversible_instance_norm
            assert isinstance(model_rin_mv.model.rin, RINorm)
            assert model_rin_mv.model.rin.input_dim == self.multivariate_series.n_components

        def helper_equality_encoders(self, first_encoders: Dict[str, Any], second_encoders: Dict[str, Any]):
            if False:
                while True:
                    i = 10
            if first_encoders is None:
                first_encoders = {}
            if second_encoders is None:
                second_encoders = {}
            assert {k: v for (k, v) in first_encoders.items() if k != 'transformer'} == {k: v for (k, v) in second_encoders.items() if k != 'transformer'}

        def helper_equality_encoders_transfo(self, first_encoders: Dict[str, Any], second_encoders: Dict[str, Any]):
            if False:
                return 10
            if first_encoders is None:
                first_encoders = {}
            if second_encoders is None:
                second_encoders = {}
            assert first_encoders.get('transformer', None).__class__ == second_encoders.get('transformer', None).__class__

        def helper_create_RNNModel(self, model_name: str, tmpdir_fn):
            if False:
                i = 10
                return i + 15
            return RNNModel(input_chunk_length=4, hidden_dim=3, add_encoders={'cyclic': {'past': ['month']}, 'datetime_attribute': {'past': ['hour']}, 'transformer': Scaler()}, n_epochs=2, model_name=model_name, work_dir=tmpdir_fn, force_reset=True, save_checkpoints=True, **tfm_kwargs)

        def helper_create_DLinearModel(self, work_dir: Optional[str]=None, model_name: str='unitest_model', add_encoders: Optional[Dict]=None, save_checkpoints: bool=False, likelihood: Optional[Likelihood]=None):
            if False:
                while True:
                    i = 10
            return DLinearModel(input_chunk_length=4, output_chunk_length=1, model_name=model_name, add_encoders=add_encoders, work_dir=work_dir, save_checkpoints=save_checkpoints, random_state=42, force_reset=True, n_epochs=1, likelihood=likelihood, **tfm_kwargs)