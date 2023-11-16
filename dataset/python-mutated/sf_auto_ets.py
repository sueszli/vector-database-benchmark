"""
StatsForecastAutoETS
-----------
"""
from typing import Optional
from statsforecast.models import AutoETS as SFAutoETS
from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.components.statsforecast_utils import create_normal_samples, one_sigma_rule, unpack_sf_dict
from darts.models.forecasting.forecasting_model import FutureCovariatesLocalForecastingModel

class StatsForecastAutoETS(FutureCovariatesLocalForecastingModel):

    def __init__(self, *autoets_args, add_encoders: Optional[dict]=None, **autoets_kwargs):
        if False:
            return 10
        'ETS based on `Statsforecasts package\n        <https://github.com/Nixtla/statsforecast>`_.\n\n        This implementation can perform faster than the :class:`ExponentialSmoothing` model,\n        but typically requires more time on the first call, because it relies\n        on Numba and jit compilation.\n\n        We refer to the `statsforecast AutoETS documentation\n        <https://nixtla.github.io/statsforecast/src/core/models.html#autoets>`_\n        for the exhaustive documentation of the arguments.\n\n        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first\n        regressing the series against the future covariates using the :class:\'LinearRegressionModel\' model and then\n        running StatsForecast\'s AutoETS on the in-sample residuals from this original regression. This approach was\n        inspired by \'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>\'_.\n\n\n        Parameters\n        ----------\n        autoets_args\n            Positional arguments for ``statsforecasts.models.AutoETS``.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        autoets_kwargs\n            Keyword arguments for ``statsforecasts.models.AutoETS``.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import StatsForecastAutoETS\n        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries\n        >>> series = AirPassengersDataset().load()\n        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series\n        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)\n        >>> # define StatsForecastAutoETS parameters\n        >>> model = StatsForecastAutoETS(season_length=12, model="AZZ")\n        >>> model.fit(series, future_covariates=future_cov)\n        >>> pred = model.predict(6, future_covariates=future_cov)\n        >>> pred.values()\n        array([[441.40323676],\n               [415.09871431],\n               [448.90785391],\n               [491.38584654],\n               [493.11817462],\n               [549.88974472]])\n        '
        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoETS(*autoets_args, **autoets_kwargs)
        self._linreg = None

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            return 10
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series
        if future_covariates is not None:
            linreg = LinearRegressionModel(lags_future_covariates=[0])
            linreg.fit(series, future_covariates=future_covariates)
            fitted_values = linreg.model.predict(X=future_covariates.slice_intersect(series).values(copy=False))
            fitted_values_ts = TimeSeries.from_times_and_values(times=series.time_index, values=fitted_values)
            resids = series - fitted_values_ts
            self._linreg = linreg
            target = resids
        else:
            target = series
        self.model.fit(target.values(copy=False).flatten())
        return self

    def _predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False):
        if False:
            while True:
                i = 10
        super()._predict(n, future_covariates, num_samples)
        forecast_dict = self.model.predict(h=n, level=[one_sigma_rule])
        (mu_ets, std) = unpack_sf_dict(forecast_dict)
        if future_covariates is not None:
            mu_linreg = self._linreg.predict(n, future_covariates=future_covariates)
            mu_linreg_values = mu_linreg.values(copy=False).reshape(n)
            mu = mu_ets + mu_linreg_values
        else:
            mu = mu_ets
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
        else:
            samples = mu
        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    @property
    def min_train_series_length(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 10

    @property
    def _supports_range_index(self) -> bool:
        if False:
            print('Hello World!')
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return True