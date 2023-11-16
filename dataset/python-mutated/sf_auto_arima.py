"""
StatsForecastAutoARIMA
-----------
"""
from typing import Optional
from statsforecast.models import AutoARIMA as SFAutoARIMA
from darts import TimeSeries
from darts.models.components.statsforecast_utils import create_normal_samples, one_sigma_rule, unpack_sf_dict
from darts.models.forecasting.forecasting_model import FutureCovariatesLocalForecastingModel

class StatsForecastAutoARIMA(FutureCovariatesLocalForecastingModel):

    def __init__(self, *autoarima_args, add_encoders: Optional[dict]=None, **autoarima_kwargs):
        if False:
            print('Hello World!')
        'Auto-ARIMA based on `Statsforecasts package\n        <https://github.com/Nixtla/statsforecast>`_.\n\n        This implementation can perform faster than the :class:`AutoARIMA` model,\n        but typically requires more time on the first call, because it relies\n        on Numba and jit compilation.\n\n        It is probabilistic, whereas :class:`AutoARIMA` is not.\n\n        We refer to the `statsforecast AutoARIMA documentation\n        <https://nixtla.github.io/statsforecast/src/core/models.html#autoarima>`_\n        for the exhaustive documentation of the arguments.\n\n        Parameters\n        ----------\n        autoarima_args\n            Positional arguments for ``statsforecasts.models.AutoARIMA``.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        autoarima_kwargs\n            Keyword arguments for ``statsforecasts.models.AutoARIMA``.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import StatsForecastAutoARIMA\n        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries\n        >>> series = AirPassengersDataset().load()\n        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series\n        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)\n        >>> # define StatsForecastAutoARIMA parameters\n        >>> model = StatsForecastAutoARIMA(season_length=12)\n        >>> model.fit(series, future_covariates=future_cov)\n        >>> pred = model.predict(6, future_covariates=future_cov)\n        >>> pred.values()\n        array([[450.55179949],\n               [415.00597806],\n               [454.61353249],\n               [486.51218795],\n               [504.09229632],\n               [555.06463942]])\n        '
        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            print('Hello World!')
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(series.values(copy=False).flatten(), X=future_covariates.values(copy=False) if future_covariates else None)
        return self

    def _predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super()._predict(n, future_covariates, num_samples)
        forecast_dict = self.model.predict(h=n, X=future_covariates.values(copy=False) if future_covariates else None, level=(one_sigma_rule,))
        (mu, std) = unpack_sf_dict(forecast_dict)
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
        else:
            samples = mu
        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    @property
    def min_train_series_length(self) -> int:
        if False:
            return 10
        return 10

    @property
    def _supports_range_index(self) -> bool:
        if False:
            return 10
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return True