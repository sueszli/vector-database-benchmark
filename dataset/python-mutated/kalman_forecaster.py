"""
Kalman Filter Forecaster
------------------------

A model producing stochastic forecasts based on the Kalman filter.
The filter is first optionally fitted on the series (using the N4SID
identification algorithm), and then run on future time steps in order
to obtain forecasts.

This implementation accepts an optional control signal (future covariates).
"""
from typing import Optional
import numpy as np
from nfoursid.kalman import Kalman
from darts.logging import get_logger
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.forecasting.forecasting_model import TransferableFutureCovariatesLocalForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class KalmanForecaster(TransferableFutureCovariatesLocalForecastingModel):

    def __init__(self, dim_x: int=1, kf: Optional[Kalman]=None, add_encoders: Optional[dict]=None):
        if False:
            i = 10
            return i + 15
        'Kalman filter Forecaster\n\n        This model uses a Kalman filter to produce forecasts. It uses a\n        :class:`darts.models.filtering.kalman_filter.KalmanFilter` object\n        and treats future values as missing values.\n\n        The model can optionally receive a :class:`nfoursid.kalman.Kalman`\n        object specifying the Kalman filter, or, if not specified, the filter\n        will be trained using the N4SID system identification algorithm.\n\n        Parameters\n        ----------\n        dim_x : int\n            Size of the Kalman filter state vector.\n        kf : nfoursid.kalman.Kalman\n            Optionally, an instance of `nfoursid.kalman.Kalman`.\n            If this is provided, the parameter dim_x is ignored. This instance will be copied for every\n            call to `predict()`, so the state is not carried over from one time series to another across several\n            calls to `predict()`.\n            The various dimensionalities of the filter must match those of the `TimeSeries` used when\n            calling `predict()`.\n            If this is specified, it is still necessary to call `fit()` before calling `predict()`,\n            although this will have no effect on the Kalman filter.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import KalmanForecaster\n        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries\n        >>> series = AirPassengersDataset().load()\n        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series\n        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)\n        >>> # increasing the size of the state vector\n        >>> model = KalmanForecaster(dim_x=12)\n        >>> model.fit(series, future_covariates=future_cov)\n        >>> pred = model.predict(6, future_covariates=future_cov)\n        >>> pred.values()\n        array([[474.40680728],\n               [440.51801726],\n               [461.94512461],\n               [494.42090089],\n               [528.6436328 ],\n               [590.30647185]])\n\n        .. note::\n            `Kalman example notebook <https://unit8co.github.io/darts/examples/10-Kalman-filter-examples.html>`_\n            presents techniques that can be used to improve the forecasts quality compared to this simple usage\n            example.\n        '
        super().__init__(add_encoders=add_encoders)
        self.dim_x = dim_x
        self.kf = kf
        self.darts_kf = KalmanFilter(dim_x, kf)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            print('Hello World!')
        super()._fit(series, future_covariates)
        if self.kf is None:
            self.darts_kf.fit(series=series, covariates=future_covariates)
        return self

    def predict(self, n: int, series: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, **kwargs) -> TimeSeries:
        if False:
            while True:
                i = 10
        series = series if series is not None else self.training_series
        return super().predict(n, series, future_covariates, num_samples, **kwargs)

    def _predict(self, n: int, series: Optional[TimeSeries]=None, historic_future_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        super()._predict(n, series, historic_future_covariates, future_covariates, num_samples)
        time_index = self._generate_new_dates(n, input_series=series)
        placeholder_vals = np.zeros((n, self.training_series.width)) * np.nan
        series_future = TimeSeries.from_times_and_values(time_index, placeholder_vals, columns=self.training_series.columns, static_covariates=self.training_series.static_covariates, hierarchy=self.training_series.hierarchy)
        series = series.append(series_future)
        if historic_future_covariates is not None:
            future_covariates = historic_future_covariates.append(future_covariates)
        filtered_series = self.darts_kf.filter(series=series, covariates=future_covariates, num_samples=num_samples)
        return filtered_series[-n:]

    @property
    def supports_multivariate(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return True