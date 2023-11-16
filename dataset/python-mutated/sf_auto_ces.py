"""
StatsForecastAutoCES
-----------
"""
from statsforecast.models import AutoCES as SFAutoCES
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import LocalForecastingModel

class StatsForecastAutoCES(LocalForecastingModel):

    def __init__(self, *autoces_args, **autoces_kwargs):
        if False:
            while True:
                i = 10
        'Auto-CES based on `Statsforecasts package\n        <https://github.com/Nixtla/statsforecast>`_.\n\n        Automatically selects the best Complex Exponential Smoothing model using an information criterion.\n        <https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074>\n\n        We refer to the `statsforecast AutoCES documentation\n        <https://nixtla.github.io/statsforecast/src/core/models.html#autoces>`_\n        for the exhaustive documentation of the arguments.\n\n        Parameters\n        ----------\n        autoces_args\n            Positional arguments for ``statsforecasts.models.AutoCES``.\n        autoces_kwargs\n            Keyword arguments for ``statsforecasts.models.AutoCES``.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import StatsForecastAutoCES\n        >>> series = AirPassengersDataset().load()\n        >>> # define StatsForecastAutoCES parameters\n        >>> model = StatsForecastAutoCES(season_length=12, model="Z")\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[453.03417969],\n               [429.34039307],\n               [488.64471436],\n               [500.28955078],\n               [519.79962158],\n               [586.47503662]])\n        '
        super().__init__()
        self.model = SFAutoCES(*autoces_args, **autoces_kwargs)

    def fit(self, series: TimeSeries):
        if False:
            for i in range(10):
                print('nop')
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(series.values(copy=False).flatten())
        return self

    def predict(self, n: int, num_samples: int=1, verbose: bool=False):
        if False:
            print('Hello World!')
        super().predict(n, num_samples)
        forecast_dict = self.model.predict(h=n)
        mu = forecast_dict['mean']
        return self._build_forecast_series(mu)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    @property
    def min_train_series_length(self) -> int:
        if False:
            while True:
                i = 10
        return 10

    @property
    def _supports_range_index(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return False