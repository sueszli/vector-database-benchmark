import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.logging import get_logger
from darts.models import ExponentialSmoothing, LinearRegressionModel, NaiveDrift, NaiveEnsembleModel, NaiveSeasonal, StatsForecastAutoARIMA, Theta
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg
logger = get_logger(__name__)
try:
    from darts.models import DLinearModel, NBEATSModel, RNNModel, TCNModel
    from darts.utils.likelihood_models import QuantileRegression
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not installed - Some ensemble models tests will be skipped.')
    TORCH_AVAILABLE = False

def _make_ts(start_value=0, n=100):
    if False:
        while True:
            i = 10
    times = pd.date_range(start='1/1/2013', periods=n, freq='D')
    pd_series = pd.Series(range(start_value, start_value + n), index=times)
    return TimeSeries.from_series(pd_series)

class TestEnsembleModels:
    series1 = tg.sine_timeseries(value_frequency=1 / 5, value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)
    seq1 = [_make_ts(0), _make_ts(10), _make_ts(20)]
    cov1 = [_make_ts(5), _make_ts(15), _make_ts(25)]

    def test_untrained_models(self):
        if False:
            while True:
                i = 10
        model = NaiveDrift()
        _ = NaiveEnsembleModel([model])
        model.fit(self.series1)
        with pytest.raises(ValueError):
            NaiveEnsembleModel([model])
        model_ens = NaiveEnsembleModel([NaiveDrift()])
        model_ens.fit(self.series1)
        assert model_ens.forecasting_models[0]._fit_called
        new_model = model_ens.untrained_model()
        assert not new_model.forecasting_models[0]._fit_called

    def test_trained_models(self):
        if False:
            for i in range(10):
                print('nop')
        'EnsembleModels can be instantiated with pre-trained GlobalForecastingModels'
        local_model = NaiveDrift()
        global_model = LinearRegressionModel(lags=2)
        local_model.fit(self.series1)
        global_model.fit(self.series1)
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model, global_model])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model.untrained_model(), global_model])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([local_model, global_model.untrained_model()])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([global_model, global_model.untrained_model()])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([global_model, global_model], train_forecasting_models=True)
        model_ens_retrain = NaiveEnsembleModel([global_model.untrained_model(), global_model.untrained_model()], train_forecasting_models=True)
        with pytest.raises(ValueError):
            model_ens_retrain.predict(1, series=self.series1)
        model_ens_retrain.fit(self.series1)
        model_ens_retrain.predict(1, series=self.series1)
        model_ens_no_retrain = NaiveEnsembleModel([global_model, global_model], train_forecasting_models=False)
        model_ens_no_retrain.predict(1, series=self.series1)

    def test_extreme_lag_inference(self):
        if False:
            print('Hello World!')
        ensemble = NaiveEnsembleModel([NaiveDrift()])
        assert ensemble.extreme_lags == (-3, -1, None, None, None, None)
        model1 = LinearRegressionModel(lags=3, lags_past_covariates=[-3, -5], lags_future_covariates=[7, 8])
        model2 = LinearRegressionModel(lags=5, lags_past_covariates=6, lags_future_covariates=[6, 9])
        ensemble = NaiveEnsembleModel([model1, model2])
        expected = (-5, 0, -6, -1, 6, 9)
        assert expected == ensemble.extreme_lags

    def test_input_models_local_models(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            NaiveEnsembleModel([])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        with pytest.raises(ValueError):
            NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()])
        NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])

    def test_call_predict_local_models(self):
        if False:
            i = 10
            return i + 15
        naive_ensemble = NaiveEnsembleModel([NaiveSeasonal(), Theta()])
        with pytest.raises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        pred1 = naive_ensemble.predict(5)
        assert self.series1.components == pred1.components

    def test_call_backtest_naive_ensemble_local_models(self):
        if False:
            i = 10
            return i + 15
        ensemble = NaiveEnsembleModel([NaiveSeasonal(5), Theta(2, 5)])
        ensemble.fit(self.series1)
        assert ensemble.extreme_lags == (-10, -1, None, None, None, None)
        ensemble.backtest(self.series1)

    def test_predict_univariate_ensemble_local_models(self):
        if False:
            for i in range(10):
                print('nop')
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_ensemble: NaiveEnsembleModel = NaiveEnsembleModel([naive, theta])
        naive_ensemble.fit(self.series1 + self.series2)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)
        np.testing.assert_array_equal(forecast_naive_ensemble.values(), forecast_mean.values())

    def test_predict_multivariate_ensemble_local_models(self):
        if False:
            for i in range(10):
                print('nop')
        multivariate_series = self.series1.stack(self.series2)
        seasonal1 = NaiveSeasonal(K=5)
        seasonal2 = NaiveSeasonal(K=8)
        naive_ensemble: NaiveEnsembleModel = NaiveEnsembleModel([seasonal1, seasonal2])
        naive_ensemble.fit(multivariate_series)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        seasonal1.fit(multivariate_series)
        seasonal2.fit(multivariate_series)
        forecast_mean = 0.5 * seasonal1.predict(5) + 0.5 * seasonal2.predict(5)
        np.testing.assert_array_equal(forecast_naive_ensemble.values(), forecast_mean.values())
        assert all(forecast_naive_ensemble.components == multivariate_series.components)

    def test_stochastic_naive_ensemble(self):
        if False:
            for i in range(10):
                print('nop')
        num_samples = 100
        model_proba_1 = LinearRegressionModel(lags=1, likelihood='quantile', random_state=42)
        model_proba_2 = LinearRegressionModel(lags=2, likelihood='quantile', random_state=42)
        naive_ensemble_proba = NaiveEnsembleModel([model_proba_1, model_proba_2])
        assert naive_ensemble_proba._is_probabilistic
        naive_ensemble_proba.fit(self.series1 + self.series2)
        pred_proba_1_sample = naive_ensemble_proba.predict(n=5)
        assert pred_proba_1_sample.n_samples == 1
        pred_proba_many_sample = naive_ensemble_proba.predict(n=5, num_samples=num_samples)
        assert pred_proba_many_sample.n_samples == num_samples
        model_alone_1 = LinearRegressionModel(lags=1, likelihood='quantile', random_state=42)
        model_alone_2 = LinearRegressionModel(lags=2, likelihood='quantile', random_state=42)
        model_alone_1.fit(self.series1 + self.series2)
        model_alone_2.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * model_alone_1.predict(5, num_samples=num_samples) + 0.5 * model_alone_2.predict(5, num_samples=num_samples)
        assert forecast_mean.values().shape == pred_proba_many_sample.values().shape
        assert forecast_mean.n_samples == pred_proba_many_sample.n_samples
        np.testing.assert_array_equal(pred_proba_many_sample.values(), forecast_mean.values())

    def test_predict_likelihood_parameters_wrong_args(self):
        if False:
            i = 10
            return i + 15
        m_deterministic = LinearRegressionModel(lags=2, output_chunk_length=2)
        m_proba_quantile1 = LinearRegressionModel(lags=2, output_chunk_length=2, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_quantile2 = LinearRegressionModel(lags=3, output_chunk_length=3, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_poisson = LinearRegressionModel(lags=2, output_chunk_length=2, likelihood='poisson')
        naive_ensemble = NaiveEnsembleModel([m_deterministic, m_proba_quantile1])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1.untrained_model(), m_proba_poisson])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1.untrained_model(), m_proba_quantile2])
        naive_ensemble.fit(self.series1 + self.series2)
        with pytest.raises(ValueError):
            naive_ensemble.predict(n=4, predict_likelihood_parameters=True)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_predict_likelihood_parameters_univariate_naive_ensemble(self):
        if False:
            i = 10
            return i + 15
        m_proba_quantile1 = LinearRegressionModel(lags=2, output_chunk_length=2, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_quantile2 = LinearRegressionModel(lags=3, output_chunk_length=2, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_quantile3 = DLinearModel(input_chunk_length=4, output_chunk_length=2, likelihood=QuantileRegression([0.05, 0.5, 0.95]), **tfm_kwargs)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1, m_proba_quantile2])
        naive_ensemble.fit(self.series1)
        pred_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile2.untrained_model(), m_proba_quantile3.untrained_model()])
        naive_ensemble.fit(self.series1)
        pred_mix_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        assert pred_ens.time_index == pred_mix_ens.time_index
        assert all(pred_ens.components == pred_mix_ens.components)
        assert pred_ens['sine_q0.05'].values() < pred_ens['sine_q0.50'].values() < pred_ens['sine_q0.95'].values()

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_predict_likelihood_parameters_multivariate_naive_ensemble(self):
        if False:
            print('Hello World!')
        m_proba_quantile1 = LinearRegressionModel(lags=2, output_chunk_length=2, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_quantile2 = LinearRegressionModel(lags=3, output_chunk_length=2, likelihood='quantile', quantiles=[0.05, 0.5, 0.95])
        m_proba_quantile3 = DLinearModel(input_chunk_length=4, output_chunk_length=2, likelihood=QuantileRegression([0.05, 0.5, 0.95]), **tfm_kwargs)
        multivariate_series = self.series1.stack(self.series2)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile1, m_proba_quantile2])
        naive_ensemble.fit(multivariate_series)
        pred_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        naive_ensemble = NaiveEnsembleModel([m_proba_quantile2.untrained_model(), m_proba_quantile3.untrained_model()])
        naive_ensemble.fit(multivariate_series)
        pred_mix_ens = naive_ensemble.predict(n=1, predict_likelihood_parameters=True)
        assert pred_ens.time_index == pred_mix_ens.time_index
        assert all(pred_ens.components == ['sine_q0.05', 'sine_q0.50', 'sine_q0.95', 'linear_q0.05', 'linear_q0.50', 'linear_q0.95'])
        assert all(pred_ens.components == pred_mix_ens.components)
        assert pred_ens['sine_q0.05'].values() < pred_ens['sine_q0.50'].values() < pred_ens['sine_q0.95'].values()
        assert pred_ens['linear_q0.05'].values() < pred_ens['linear_q0.50'].values() < pred_ens['linear_q0.95'].values()

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_input_models_global_models(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel])
        NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel(10, 2)])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_call_predict_global_models_univariate_input_no_covariates(self):
        if False:
            while True:
                i = 10
        naive_ensemble = NaiveEnsembleModel([RNNModel(12, n_epochs=1, **tfm_kwargs), TCNModel(10, 2, n_epochs=1, **tfm_kwargs), NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs)])
        with pytest.raises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_call_predict_global_models_multivariate_input_no_covariates(self):
        if False:
            print('Hello World!')
        naive_ensemble = NaiveEnsembleModel([RNNModel(12, n_epochs=1, **tfm_kwargs), TCNModel(10, 2, n_epochs=1, **tfm_kwargs), NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs)])
        naive_ensemble.fit(self.seq1)
        naive_ensemble.predict(n=5, series=self.seq1)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_call_predict_global_models_multivariate_input_with_covariates(self):
        if False:
            print('Hello World!')
        naive_ensemble = NaiveEnsembleModel([RNNModel(12, n_epochs=1, **tfm_kwargs), TCNModel(10, 2, n_epochs=1, **tfm_kwargs), NBEATSModel(10, 2, n_epochs=1, **tfm_kwargs)])
        naive_ensemble.fit(self.seq1, self.cov1)
        predict_series = [s[:12] for s in self.seq1]
        predict_covariates = [c[:14] for c in self.cov1]
        naive_ensemble.predict(n=2, series=predict_series, past_covariates=predict_covariates)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_input_models_mixed(self):
        if False:
            return 10
        naive_ensemble = NaiveEnsembleModel([NaiveDrift(), RNNModel(12, n_epochs=1, **tfm_kwargs)])
        assert not naive_ensemble.is_local_ensemble
        assert not naive_ensemble.is_global_ensemble
        with pytest.raises(ValueError):
            naive_ensemble.fit([self.series1, self.series2])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='requires torch')
    def test_call_predict_different_covariates_support(self):
        if False:
            print('Hello World!')
        local_ensemble_one_covs = NaiveEnsembleModel([NaiveDrift(), StatsForecastAutoARIMA()])
        with pytest.raises(ValueError):
            local_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        local_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)
        mixed_ensemble_one_covs = NaiveEnsembleModel([NaiveDrift(), RNNModel(12, n_epochs=1, **tfm_kwargs)])
        with pytest.raises(ValueError):
            mixed_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        mixed_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)
        mixed_ensemble_future_covs = NaiveEnsembleModel([StatsForecastAutoARIMA(), RNNModel(12, n_epochs=1, **tfm_kwargs)])
        mixed_ensemble_future_covs.fit(self.series1, future_covariates=self.series2)
        with pytest.raises(ValueError):
            mixed_ensemble_future_covs.fit(self.series1, past_covariates=self.series2)
        global_ensemble_both_covs = NaiveEnsembleModel([LinearRegressionModel(lags=1, lags_past_covariates=[-1]), LinearRegressionModel(lags=1, lags_future_covariates=[1])])
        with pytest.raises(ValueError):
            global_ensemble_both_covs.fit(self.series1, past_covariates=self.series2)
        with pytest.raises(ValueError):
            global_ensemble_both_covs.fit(self.series1, future_covariates=self.series2)
        global_ensemble_both_covs.fit(self.series1, past_covariates=self.series2, future_covariates=self.series2)

    def test_fit_multivar_ts_with_local_models(self):
        if False:
            for i in range(10):
                print('nop')
        naive = NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])
        with pytest.raises(ValueError):
            naive.fit(self.seq1)

    def test_fit_univar_ts_with_covariates_for_local_models(self):
        if False:
            i = 10
            return i + 15
        naive = NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])
        with pytest.raises(ValueError):
            naive.fit(self.series1, self.series2)

    def test_predict_with_target(self):
        if False:
            print('Hello World!')
        series_long = self.series1
        series_short = series_long[:25]
        ensemble_model = self.get_global_ensemble_model()
        ensemble_model.fit(series_short, past_covariates=series_long)
        preds = ensemble_model.predict(n=5, past_covariates=series_long)
        assert isinstance(preds, TimeSeries)
        preds = ensemble_model.predict(n=5, series=series_long, past_covariates=series_long)
        assert isinstance(preds, TimeSeries)
        preds = ensemble_model.predict(n=5, series=[series_long] * 2, past_covariates=[series_long] * 2)
        assert isinstance(preds, list) and len(preds) == 2
        preds = ensemble_model.predict(n=5, series=[series_long], past_covariates=[series_long])
        assert isinstance(preds, list) and len(preds) == 1
        ensemble_model = self.get_global_ensemble_model()
        ensemble_model.fit([series_short] * 2, past_covariates=[series_long] * 2)
        with pytest.raises(ValueError):
            ensemble_model.predict(n=5, past_covariates=series_long)
        preds = ensemble_model.predict(n=5, series=series_long, past_covariates=series_long)
        assert isinstance(preds, TimeSeries)
        preds = ensemble_model.predict(n=5, series=[series_long] * 2, past_covariates=[series_long] * 2)
        assert isinstance(preds, list) and len(preds) == 2
        preds = ensemble_model.predict(n=5, series=[series_long], past_covariates=[series_long])
        assert isinstance(preds, list) and len(preds) == 1

    @staticmethod
    def get_global_ensemble_model(output_chunk_length=5):
        if False:
            while True:
                i = 10
        lags = [-1, -2, -5]
        return NaiveEnsembleModel(forecasting_models=[LinearRegressionModel(lags=lags, lags_past_covariates=lags, output_chunk_length=output_chunk_length), LinearRegressionModel(lags=lags, lags_past_covariates=lags, output_chunk_length=output_chunk_length)])