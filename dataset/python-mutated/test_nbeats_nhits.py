import numpy as np
import pytest
from darts.logging import get_logger
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg
logger = get_logger(__name__)
try:
    from darts.models.forecasting.nbeats import NBEATSModel
    from darts.models.forecasting.nhits import NHiTSModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. Nbeats and NHiTs tests will be skipped.')
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:

    class TestNbeatsNhitsModel:

        def test_creation(self):
            if False:
                print('Hello World!')
            with pytest.raises(ValueError):
                NBEATSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=3, layer_widths=[1, 2])
            with pytest.raises(ValueError):
                NHiTSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=3, layer_widths=[1, 2])

        def test_fit(self):
            if False:
                while True:
                    i = 10
            large_ts = tg.constant_timeseries(length=100, value=1000)
            small_ts = tg.constant_timeseries(length=100, value=10)
            for model_cls in [NBEATSModel, NHiTSModel]:
                model = model_cls(input_chunk_length=1, output_chunk_length=1, n_epochs=10, num_stacks=1, num_blocks=1, layer_widths=20, random_state=42, **tfm_kwargs)
                model.fit(large_ts[:98])
                pred = model.predict(n=2).values()[0]
                model2 = model_cls(input_chunk_length=1, output_chunk_length=1, n_epochs=10, num_stacks=1, num_blocks=1, layer_widths=20, random_state=42, **tfm_kwargs)
                model2.fit(small_ts[:98])
                pred2 = model2.predict(n=2).values()[0]
                assert abs(pred2 - 10) < abs(pred - 10)
                pred3 = model2.predict(n=1)
                assert len(pred3) == 1

        def test_multivariate(self):
            if False:
                while True:
                    i = 10
            series_multivariate = tg.linear_timeseries(length=100).stack(tg.linear_timeseries(length=100, start_value=0, end_value=0.5))
            for model_cls in [NBEATSModel, NHiTSModel]:
                model = model_cls(input_chunk_length=3, output_chunk_length=1, n_epochs=20, random_state=42, **tfm_kwargs)
                model.fit(series_multivariate)
                res = model.predict(n=2).values()
                assert abs(np.average(res - np.array([[1.01, 1.02], [0.505, 0.51]])) < 0.03)
                series_covariates = tg.linear_timeseries(length=100).stack(tg.linear_timeseries(length=100, start_value=0, end_value=0.1))
                model = model_cls(input_chunk_length=3, output_chunk_length=4, n_epochs=5, random_state=42, **tfm_kwargs)
                model.fit(series_multivariate, past_covariates=series_covariates)
                res = model.predict(n=3, series=series_multivariate, past_covariates=series_covariates).values()
                assert len(res) == 3
                assert abs(np.average(res)) < 5

        def test_nhits_sampling_sizes(self):
            if False:
                return 10
            with pytest.raises(ValueError):
                NHiTSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=1, num_blocks=2, pooling_kernel_sizes=((1,), (1,)), n_freq_downsample=((1,), (1,)))
            with pytest.raises(ValueError):
                NHiTSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=2, num_blocks=2, pooling_kernel_sizes=((1, 1), (1, 1)), n_freq_downsample=((2, 1), (2, 2)))
            _ = NHiTSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=2, num_blocks=2, pooling_kernel_sizes=((2, 1), (2, 1)), n_freq_downsample=((2, 1), (2, 1)))
            model = NHiTSModel(input_chunk_length=1, output_chunk_length=1, num_stacks=2, num_blocks=2)
            assert model.n_freq_downsample[-1][-1] == 1

        def test_logtensorboard(self, tmpdir_module):
            if False:
                for i in range(10):
                    print('nop')
            ts = tg.constant_timeseries(length=50, value=10)
            architectures = [True, False]
            for architecture in architectures:
                model = NBEATSModel(input_chunk_length=1, output_chunk_length=1, n_epochs=1, log_tensorboard=True, work_dir=tmpdir_module, generic_architecture=architecture, pl_trainer_kwargs={'log_every_n_steps': 1, **tfm_kwargs['pl_trainer_kwargs']})
                model.fit(ts)
                model.predict(n=2)

        def test_activation_fns(self):
            if False:
                print('Hello World!')
            ts = tg.constant_timeseries(length=50, value=10)
            for model_cls in [NBEATSModel, NHiTSModel]:
                model = model_cls(input_chunk_length=1, output_chunk_length=1, n_epochs=10, num_stacks=1, num_blocks=1, layer_widths=20, random_state=42, activation='LeakyReLU', **tfm_kwargs)
                model.fit(ts)
                with pytest.raises(ValueError):
                    model = model_cls(input_chunk_length=1, output_chunk_length=1, n_epochs=10, num_stacks=1, num_blocks=1, layer_widths=20, random_state=42, activation='invalid', **tfm_kwargs)
                    model.fit(ts)