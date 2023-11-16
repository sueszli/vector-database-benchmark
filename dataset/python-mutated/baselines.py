"""
Baseline Models
---------------

A collection of simple benchmark models for univariate series.
"""
from typing import List, Optional, Sequence, Union
import numpy as np
from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel, LocalForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class NaiveMean(LocalForecastingModel):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Naive Mean Model\n\n        This model has no parameter, and always predicts the\n        mean value of the training series.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import NaiveMean\n        >>> series = AirPassengersDataset().load()\n        >>> model = NaiveMean()\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[280.29861111],\n              [280.29861111],\n              [280.29861111],\n              [280.29861111],\n              [280.29861111],\n              [280.29861111]])\n        '
        super().__init__()
        self.mean_val = None

    @property
    def supports_multivariate(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def fit(self, series: TimeSeries):
        if False:
            return 10
        super().fit(series)
        self.mean_val = np.mean(series.values(copy=False), axis=0)
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            return 10
        super().predict(n, num_samples)
        forecast = np.tile(self.mean_val, (n, 1))
        return self._build_forecast_series(forecast)

class NaiveSeasonal(LocalForecastingModel):

    def __init__(self, K: int=1):
        if False:
            return 10
        'Naive Seasonal Model\n\n        This model always predicts the value of `K` time steps ago.\n        When `K=1`, this model predicts the last value of the training set.\n        When `K>1`, it repeats the last `K` values of the training set.\n\n        Parameters\n        ----------\n        K\n            the number of last time steps of the training set to repeat\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import NaiveSeasonal\n        >>> series = AirPassengersDataset().load()\n        # prior analysis suggested seasonality of 12\n        >>> model = NaiveSeasonal(K=12)\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[417.],\n               [391.],\n               [419.],\n               [461.],\n               [472.],\n               [535.]])\n        '
        super().__init__()
        self.last_k_vals = None
        self.K = K

    @property
    def supports_multivariate(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @property
    def min_train_series_length(self):
        if False:
            i = 10
            return i + 15
        return max(self.K, 3)

    def fit(self, series: TimeSeries):
        if False:
            print('Hello World!')
        super().fit(series)
        raise_if_not(len(series) >= self.K, f'The time series requires at least K={self.K} points', logger)
        self.last_k_vals = series.values(copy=False)[-self.K:, :]
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            while True:
                i = 10
        super().predict(n, num_samples)
        forecast = np.array([self.last_k_vals[i % self.K, :] for i in range(n)])
        return self._build_forecast_series(forecast)

class NaiveDrift(LocalForecastingModel):

    def __init__(self):
        if False:
            print('Hello World!')
        'Naive Drift Model\n\n        This model fits a line between the first and last point of the training series,\n        and extends it in the future. For a training series of length :math:`T`, we have:\n\n        .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import NaiveDrift\n        >>> series = AirPassengersDataset().load()\n        >>> model = NaiveDrift()\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[434.23776224],\n               [436.47552448],\n               [438.71328671],\n               [440.95104895],\n               [443.18881119],\n               [445.42657343]])\n        '
        super().__init__()

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return True

    def fit(self, series: TimeSeries):
        if False:
            while True:
                i = 10
        super().fit(series)
        assert series.n_samples == 1, 'This model expects deterministic time series'
        series = self.training_series
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            print('Hello World!')
        super().predict(n, num_samples)
        (first, last) = (self.training_series.first_values(), self.training_series.last_values())
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n + 1)[1:]
        return self._build_forecast_series(forecast)

class NaiveMovingAverage(LocalForecastingModel):

    def __init__(self, input_chunk_length: int=1):
        if False:
            i = 10
            return i + 15
        'Naive Moving Average Model\n\n        This model forecasts using an auto-regressive moving average (ARMA).\n\n        Parameters\n        ----------\n        input_chunk_length\n            The size of the sliding window used to calculate the moving average\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import NaiveMovingAverage\n        >>> series = AirPassengersDataset().load()\n        # using the average of the last 6 months\n        >>> model = NaiveMovingAverage(input_chunk_length=6)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[503.16666667],\n               [483.36111111],\n               [462.9212963 ],\n               [455.40817901],\n               [454.47620885],\n               [465.22224366]])\n        '
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.rolling_window = None

    @property
    def supports_multivariate(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @property
    def min_train_series_length(self):
        if False:
            while True:
                i = 10
        return self.input_chunk_length

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'NaiveMovingAverage({self.input_chunk_length})'

    def fit(self, series: TimeSeries):
        if False:
            print('Hello World!')
        super().fit(series)
        raise_if_not(series.is_deterministic, 'This model expects deterministic time series', logger)
        self.rolling_window = series[-self.input_chunk_length:].values(copy=False)
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            return 10
        super().predict(n, num_samples)
        predictions_with_observations = np.concatenate((self.rolling_window, np.zeros(shape=(n, self.rolling_window.shape[1]))), axis=0)
        rolling_sum = sum(self.rolling_window)
        chunk_length = self.input_chunk_length
        for i in range(chunk_length, chunk_length + n):
            prediction = rolling_sum / chunk_length
            predictions_with_observations[i] = prediction
            lost_value = predictions_with_observations[i - chunk_length]
            rolling_sum += prediction - lost_value
        return self._build_forecast_series(predictions_with_observations[-n:])

class NaiveEnsembleModel(EnsembleModel):

    def __init__(self, forecasting_models: List[ForecastingModel], train_forecasting_models: bool=True, show_warnings: bool=True):
        if False:
            while True:
                i = 10
        'Naive combination model\n\n        Naive implementation of `EnsembleModel`\n        Returns the average of all predictions of the constituent models\n\n        If `future_covariates` or `past_covariates` are provided at training or inference time,\n        they will be passed only to the models supporting them.\n\n        Parameters\n        ----------\n        forecasting_models\n            List of forecasting models whose predictions to ensemble\n        train_forecasting_models\n            Whether to train the `forecasting_models` from scratch. If `False`, the models are not trained when calling\n            `fit()` and `predict()` can be called directly (only supported if all the `forecasting_models` are\n            pretrained `GlobalForecastingModels`). Default: ``True``.\n        show_warnings\n            Whether to show warnings related to models covariates support.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import NaiveEnsembleModel, NaiveSeasonal, LinearRegressionModel\n        >>> series = AirPassengersDataset().load()\n        >>> # defining the ensemble\n        >>> model = NaiveEnsembleModel([NaiveSeasonal(K=12), LinearRegressionModel(lags=4)])\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[439.23152974],\n               [431.41161602],\n               [439.72888401],\n               [453.70180806],\n               [454.96757177],\n               [485.16604194]])\n        '
        super().__init__(forecasting_models=forecasting_models, train_num_samples=1, train_samples_reduction=None, train_forecasting_models=train_forecasting_models, show_warnings=show_warnings)
        if self.all_trained and (not train_forecasting_models):
            self._fit_called = True

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None):
        if False:
            return 10
        super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        if self.train_forecasting_models:
            for model in self.forecasting_models:
                model._fit_wrapper(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        return self

    def ensemble(self, predictions: Union[TimeSeries, Sequence[TimeSeries]], series: Union[TimeSeries, Sequence[TimeSeries]], num_samples: int=1, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            print('Hello World!')
        'Average the `forecasting_models` predictions, component-wise'
        raise_if(predict_likelihood_parameters and (not self.supports_likelihood_parameter_prediction), '`predict_likelihood_parameters=True` is supported only if all the `forecasting_models` are probabilistic and fitting the same likelihood.', logger)
        if isinstance(predictions, Sequence):
            return [self._target_average(p, ts) if not predict_likelihood_parameters else self._params_average(p, ts) for (p, ts) in zip(predictions, series)]
        else:
            return self._target_average(predictions, series) if not predict_likelihood_parameters else self._params_average(predictions, series)

    def _target_average(self, prediction: TimeSeries, series: TimeSeries) -> TimeSeries:
        if False:
            print('Hello World!')
        'Average across the components, keep n_samples, rename components'
        n_forecasting_models = len(self.forecasting_models)
        n_components = series.n_components
        prediction_values = prediction.all_values(copy=False)
        target_values = np.zeros((prediction.n_timesteps, n_components, prediction.n_samples))
        for idx_target in range(n_components):
            target_values[:, idx_target] = prediction_values[:, range(idx_target, n_forecasting_models * n_components, n_components)].mean(axis=1)
        return TimeSeries.from_times_and_values(times=prediction.time_index, values=target_values, freq=series.freq, columns=series.components, static_covariates=series.static_covariates, hierarchy=series.hierarchy)

    def _params_average(self, prediction: TimeSeries, series: TimeSeries) -> TimeSeries:
        if False:
            print('Hello World!')
        'Average across the components after grouping by likelihood parameter, rename components'
        likelihood = getattr(self.forecasting_models[0], 'likelihood')
        if isinstance(likelihood, str):
            likelihood_n_params = self.forecasting_models[0].num_parameters
        else:
            likelihood_n_params = likelihood.num_parameters
        n_forecasting_models = len(self.forecasting_models)
        n_components = series.n_components
        prediction_values = prediction.values(copy=False)
        params_values = np.zeros((prediction.n_timesteps, likelihood_n_params * n_components))
        for idx_param in range(likelihood_n_params * n_components):
            params_values[:, idx_param] = prediction_values[:, range(idx_param, likelihood_n_params * n_forecasting_models * n_components, likelihood_n_params * n_components)].mean(axis=1)
        return TimeSeries.from_times_and_values(times=prediction.time_index, values=params_values, freq=series.freq, columns=prediction.components[:likelihood_n_params * n_components], static_covariates=None, hierarchy=None)