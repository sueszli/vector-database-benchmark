"""
VARIMA
-----

Models for VARIMA (Vector Autoregressive moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Vector_autoregression
"""
from typing import Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VARMAX as staVARMA
from darts.logging import get_logger, raise_if
from darts.models.forecasting.forecasting_model import TransferableFutureCovariatesLocalForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class VARIMA(TransferableFutureCovariatesLocalForecastingModel):

    def __init__(self, p: int=1, d: int=0, q: int=0, trend: Optional[str]=None, add_encoders: Optional[dict]=None):
        if False:
            return 10
        'VARIMA\n\n        Parameters\n        ----------\n        p : int\n            Order (number of time lags) of the autoregressive model (AR)\n        d : int\n            The order of differentiation; i.e., the number of times the data\n            have had past values subtracted. (I) Note that Darts only supports d <= 1 because for\n            d > 1 the optimizer often does not result in stable predictions. If results are not stable\n            for d = 1 try to set d = 0 and enable the trend parameter\n            to account for possible non-stationarity.\n        q : int\n            The size of the moving average window (MA).\n        trend: str\n            Parameter controlling the deterministic trend. \'n\' indicates no trend,\n            \'c\' a constant term, \'t\' linear trend in time, and \'ct\' includes both.\n            Default is \'c\' for models without integration, and no trend for models with integration.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n\n        Examples\n        --------\n        >>> from darts.datasets import ETTh2Dataset\n        >>> from darts.models import VARIMA\n        >>> from darts.utils.timeseries_generation import holidays_timeseries\n        >>> # forecasting the High UseFul Load ("HUFL") and Oil Temperature ("OT")\n        >>> series = ETTh2Dataset().load()[:500][["HUFL", "OT"]]\n        >>> # optionally, use some future covariates; e.g. encode each timestep whether it is on a holiday\n        >>> future_cov = holidays_timeseries(series.time_index, "CN", add_length=6)\n        >>> # no clear trend in the dataset\n        >>> model = VARIMA(trend="n")\n        >>> model.fit(series, future_covariates=future_cov)\n        >>> pred = model.predict(6, future_covariates=future_cov)\n        >>> # the two targets are predicted together\n        >>> pred.values()\n        array([[48.11846185, 47.94272629],\n               [49.85314633, 47.97713346],\n               [51.16145791, 47.99804203],\n               [52.14674087, 48.00872598],\n               [52.88729152, 48.01166578],\n               [53.44242919, 48.00874069]])\n        '
        super().__init__(add_encoders=add_encoders)
        self.p = p
        self.d = d
        self.q = q
        self.trend = trend
        self.model = None
        assert d <= 1, 'd > 1 not supported.'

    def _differentiate_series(self, series: TimeSeries) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        'Differentiate the series self.d times'
        for _ in range(self.d):
            series = TimeSeries.from_dataframe(df=series.pd_dataframe(copy=False).diff().dropna(), static_covariates=series.static_covariates, hierarchy=series.hierarchy)
        return series

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            print('Hello World!')
        self._last_values = series.last_values()
        series = self._differentiate_series(series)
        super().fit(series, future_covariates)
        return self

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None) -> None:
        if False:
            i = 10
            return i + 15
        super()._fit(series, future_covariates)
        self._assert_multivariate(series)
        self.training_historic_future_covariates = future_covariates
        m = staVARMA(endog=series.values(copy=False), exog=future_covariates.values(copy=False) if future_covariates else None, order=(self.p, self.q), trend=self.trend)
        self.model = m.fit(disp=0)

    def _predict(self, n: int, series: Optional[TimeSeries]=None, historic_future_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False) -> TimeSeries:
        if False:
            print('Hello World!')
        if num_samples > 1 and self.trend:
            logger.warning('Trends are not well supported yet for getting probabilistic forecasts with ARIMA.If you run into issues, try calling fit() with num_samples=1 or removing the trend fromyour model.')
        self._last_num_samples = num_samples
        super()._predict(n, series, historic_future_covariates, future_covariates, num_samples)
        if series is not None:
            self._training_last_values = self._last_values
            self._last_values = series.last_values()
            series = self._differentiate_series(series)
            if historic_future_covariates and self.d > 0:
                historic_future_covariates = historic_future_covariates.slice_intersect(series)
            self.model = self.model.apply(series.values(copy=False), exog=historic_future_covariates.values(copy=False) if historic_future_covariates else None)
        if num_samples == 1:
            forecast = self.model.forecast(steps=n, exog=future_covariates.values(copy=False) if future_covariates else None)
        else:
            forecast = self.model.simulate(nsimulations=n, repetitions=num_samples, initial_state=self.model.states.predicted[-1, :], exog=future_covariates.values(copy=False) if future_covariates else None)
        forecast = self._invert_transformation(forecast)
        if series is not None:
            self.model = self.model.apply(self._orig_training_series.values(copy=False), exog=self.training_historic_future_covariates.values(copy=False) if self.training_historic_future_covariates else None)
            self._last_values = self._training_last_values
        return self._build_forecast_series(np.array(forecast))

    def _invert_transformation(self, series_df: pd.DataFrame):
        if False:
            return 10
        if self.d == 0:
            return series_df
        if self._last_num_samples > 1:
            series_df = np.tile(self._last_values, (self._last_num_samples, 1)).T + series_df.cumsum(axis=0)
        else:
            series_df = self._last_values + series_df.cumsum(axis=0)
        return series_df

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return True

    @property
    def min_train_series_length(self) -> int:
        if False:
            i = 10
            return i + 15
        return 30

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            print('Hello World!')
        return True

    @property
    def _supports_range_index(self) -> bool:
        if False:
            return 10
        raise_if(self.trend and self.trend != 'c', "'trend' is not None. Range indexing is not supported in that case.", logger)
        return True