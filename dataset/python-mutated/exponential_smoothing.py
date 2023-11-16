"""
Exponential Smoothing
---------------------
"""
from typing import Any, Dict, Optional
import numpy as np
import statsmodels.tsa.holtwinters as hw
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode
logger = get_logger(__name__)

class ExponentialSmoothing(LocalForecastingModel):

    def __init__(self, trend: Optional[ModelMode]=ModelMode.ADDITIVE, damped: Optional[bool]=False, seasonal: Optional[SeasonalityMode]=SeasonalityMode.ADDITIVE, seasonal_periods: Optional[int]=None, random_state: int=0, kwargs: Optional[Dict[str, Any]]=None, **fit_kwargs):
        if False:
            return 10
        "Exponential Smoothing\n\n        This is a wrapper around\n        `statsmodels  Holt-Winters' Exponential Smoothing\n        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`_;\n        we refer to this link for the original and more complete documentation of the parameters.\n\n        `trend` must be a ``ModelMode`` Enum member. You can access the Enum with\n         ``from darts.utils.utils import ModelMode``.\n        `seasonal` must be a ``SeasonalityMode`` Enum member. You can access the Enum with\n        ``from darts.utils.utils import SeasonalityMode``.\n\n        ``ExponentialSmoothing(trend=ModelMode.NONE, seasonal=SeasonalityMode.NONE)`` corresponds to a single\n        exponential smoothing.\n        ``ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.NONE)`` corresponds to a Holt's\n        exponential smoothing.\n\n        Please note that automatic `seasonal_period` selection (setting the `seasonal_periods` parameter equal to\n        `None`) can sometimes lead to errors if the input time series is too short. In these cases we suggest to\n        manually set the `seasonal_periods` parameter to a positive integer.\n\n        Parameters\n        ----------\n        trend\n            Type of trend component. Either ``ModelMode.ADDITIVE``, ``ModelMode.MULTIPLICATIVE``, ``ModelMode.NONE``,\n            or ``None``. Defaults to ``ModelMode.ADDITIVE``.\n        damped\n            Should the trend component be damped. Defaults to False.\n        seasonal\n            Type of seasonal component. Either ``SeasonalityMode.ADDITIVE``, ``SeasonalityMode.MULTIPLICATIVE``,\n            ``SeasonalityMode.NONE``, or ``None``. Defaults to ``SeasonalityMode.ADDITIVE``.\n        seasonal_periods\n            The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily\n            data with a weekly cycle. If not set, inferred from frequency of the series.\n        kwargs\n            Some optional keyword arguments that will be used to call\n            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing()`.\n            See `the documentation\n            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`_.\n        fit_kwargs\n            Some optional keyword arguments that will be used to call\n            :func:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit()`.\n            See `the documentation\n            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html>`_.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import ExponentialSmoothing\n        >>> from darts.utils.utils import ModelMode, SeasonalityMode\n        >>> series = AirPassengersDataset().load()\n        >>> # using Holt's exponential smoothing\n        >>> model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE)\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[445.24283838],\n               [418.22618932],\n               [465.31305075],\n               [494.95129261],\n               [505.4770514 ],\n               [573.31519186]])\n        "
        super().__init__()
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.infer_seasonal_periods = seasonal_periods is None
        self.seasonal_periods = seasonal_periods
        self.constructor_kwargs = dict() if kwargs is None else kwargs
        self.fit_kwargs = fit_kwargs
        self.model = None
        np.random.seed(random_state)

    def fit(self, series: TimeSeries):
        if False:
            print('Hello World!')
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        seasonal_periods_param = None if self.infer_seasonal_periods else self.seasonal_periods
        if self.seasonal_periods is None and series.has_range_index:
            seasonal_periods_param = 12
        hw_model = hw.ExponentialSmoothing(series.values(copy=False), trend=self.trend if self.trend is None else self.trend.value, damped_trend=self.damped, seasonal=self.seasonal if self.seasonal is None else self.seasonal.value, seasonal_periods=seasonal_periods_param, freq=series.freq if series.has_datetime_index else None, dates=series.time_index if series.has_datetime_index else None, **self.constructor_kwargs)
        hw_results = hw_model.fit(**self.fit_kwargs)
        self.model = hw_results
        if self.infer_seasonal_periods:
            self.seasonal_periods = hw_model.seasonal_periods
        return self

    def predict(self, n, num_samples=1, verbose: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super().predict(n, num_samples)
        if num_samples == 1:
            forecast = self.model.forecast(n)
        else:
            forecast = np.expand_dims(self.model.simulate(n, repetitions=num_samples), axis=1)
        return self._build_forecast_series(forecast)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            print('Hello World!')
        return False

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            print('Hello World!')
        return True

    @property
    def min_train_series_length(self) -> int:
        if False:
            print('Hello World!')
        if self.seasonal_periods is not None and self.seasonal_periods > 1:
            return 2 * self.seasonal_periods
        return 3