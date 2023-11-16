"""
StatsForecastAutoTheta
-----------
"""
from statsforecast.models import AutoTheta as SFAutoTheta
from darts import TimeSeries
from darts.models.components.statsforecast_utils import create_normal_samples, one_sigma_rule, unpack_sf_dict
from darts.models.forecasting.forecasting_model import LocalForecastingModel

class StatsForecastAutoTheta(LocalForecastingModel):

    def __init__(self, *autotheta_args, **autotheta_kwargs):
        if False:
            print('Hello World!')
        'Auto-Theta based on `Statsforecasts package\n        <https://github.com/Nixtla/statsforecast>`_.\n\n        Automatically selects the best Theta (Standard Theta Model (‘STM’), Optimized Theta Model (‘OTM’),\n        Dynamic Standard Theta Model (‘DSTM’), Dynamic Optimized Theta Model (‘DOTM’)) model using mse.\n        <https://www.sciencedirect.com/science/article/pii/S0169207016300243>\n\n        It is probabilistic, whereas :class:`FourTheta` is not.\n\n        We refer to the `statsforecast AutoTheta documentation\n        <https://nixtla.github.io/statsforecast/src/core/models.html#autotheta>`_\n        for the exhaustive documentation of the arguments.\n\n        Parameters\n        ----------\n        autotheta_args\n            Positional arguments for ``statsforecasts.models.AutoTheta``.\n        autotheta_kwargs\n            Keyword arguments for ``statsforecasts.models.AutoTheta``.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import StatsForecastAutoTheta\n        >>> series = AirPassengersDataset().load()\n        >>> # define StatsForecastAutoTheta parameters\n        >>> model = StatsForecastAutoTheta(season_length=12)\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[442.94078295],\n               [432.22936898],\n               [495.30609727],\n               [482.30625563],\n               [487.49312172],\n               [555.57902659]])\n        '
        super().__init__()
        self.model = SFAutoTheta(*autotheta_args, **autotheta_kwargs)

    def fit(self, series: TimeSeries):
        if False:
            print('Hello World!')
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(series.values(copy=False).flatten())
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            return 10
        super().predict(n, num_samples)
        forecast_dict = self.model.predict(h=n, level=(one_sigma_rule,))
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
            for i in range(10):
                print('nop')
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True