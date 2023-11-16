"""
Croston method
--------------
"""
from typing import Optional
from statsforecast.models import TSB as CrostonTSB
from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA
from darts.logging import raise_if, raise_if_not
from darts.models.forecasting.forecasting_model import FutureCovariatesLocalForecastingModel
from darts.timeseries import TimeSeries

class Croston(FutureCovariatesLocalForecastingModel):

    def __init__(self, version: str='classic', alpha_d: float=None, alpha_p: float=None, add_encoders: Optional[dict]=None):
        if False:
            i = 10
            return i + 15
        'An implementation of the `Croston method\n        <https://otexts.com/fpp3/counts.html>`_ for intermittent\n        count series.\n\n        Relying on the implementation of `Statsforecasts package\n        <https://github.com/Nixtla/statsforecast>`_.\n\n        Parameters\n        ----------\n        version\n            - "classic" corresponds to classic Croston.\n            - "optimized" corresponds to optimized classic Croston, which searches\n              for the optimal ``alpha`` smoothing parameter and can take longer\n              to run. Otherwise, a fixed value of ``alpha=0.1`` is used.\n            - "sba" corresponds to the adjustment of the Croston method known as\n              the Syntetos-Boylan Approximation [1]_.\n            - "tsb" corresponds to the adjustment of the Croston method proposed by\n              Teunter, Syntetos and Babai [2]_. In this case, `alpha_d` and `alpha_p` must\n              be set.\n        alpha_d\n            For the "tsb" version, the alpha smoothing parameter to apply on demand.\n        alpha_p\n            For the "tsb" version, the alpha smoothing parameter to apply on probability.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n\n        References\n        ----------\n        .. [1] Aris A. Syntetos and John E. Boylan. The accuracy of intermittent demand estimates.\n               International Journal of Forecasting, 21(2):303 – 314, 2005.\n        .. [2] Ruud H. Teunter, Aris A. Syntetos, and M. Zied Babai.\n               Intermittent demand: Linking forecasting to inventory obsolescence.\n               European Journal of Operational Research, 214(3):606 – 615, 2011.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import Croston\n        >>> series = AirPassengersDataset().load()\n        >>> # use the optimized version to automatically select best alpha parameter\n        >>> model = Croston(version="optimized")\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[461.7666],\n               [461.7666],\n               [461.7666],\n               [461.7666],\n               [461.7666],\n               [461.7666]])\n        '
        super().__init__(add_encoders=add_encoders)
        raise_if_not(version.lower() in ['classic', 'optimized', 'sba', 'tsb'], 'The provided "version" parameter must be set to "classic", "optimized", "sba" or "tsb".')
        if version == 'classic':
            self.model = CrostonClassic()
        elif version == 'optimized':
            self.model = CrostonOptimized()
        elif version == 'sba':
            self.model = CrostonSBA()
        else:
            raise_if(alpha_d is None or alpha_p is None, 'alpha_d and alpha_p must be specified when using "tsb".')
            self.alpha_d = alpha_d
            self.alpha_p = alpha_p
            self.model = CrostonTSB(alpha_d=self.alpha_d, alpha_p=self.alpha_p)
        self.version = version

    @property
    def supports_multivariate(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            for i in range(10):
                print('nop')
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(y=series.values(copy=False).flatten(), X=future_covariates.values(copy=False).flatten() if future_covariates is not None else None)
        return self

    def _predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False):
        if False:
            i = 10
            return i + 15
        super()._predict(n, future_covariates, num_samples)
        values = self.model.predict(h=n, X=future_covariates.values(copy=False).flatten() if future_covariates is not None else None)['mean']
        return self._build_forecast_series(values)

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
            return 10
        return False