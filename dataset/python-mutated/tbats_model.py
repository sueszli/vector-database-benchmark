"""
BATS and TBATS
--------------

(T)BATS models [1]_ stand for

* (Trigonometric)
* Box-Cox
* ARMA errors
* Trend
* Seasonal components

They are appropriate to model "complex
seasonal time series such as those with multiple
seasonal periods, high frequency seasonality,
non-integer seasonality and dual-calendar effects" [1]_.

References
----------
.. [1] https://robjhyndman.com/papers/ComplexSeasonality.pdf
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.special import inv_boxcox
from tbats import BATS as tbats_BATS
from tbats import TBATS as tbats_TBATS
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

def _seasonality_from_freq(series: TimeSeries):
    if False:
        i = 10
        return i + 15
    '\n    Infer a naive seasonality based on the frequency\n    '
    if series.has_range_index:
        return None
    freq = series.freq_str
    if freq in ['B', 'C']:
        return [5]
    elif freq == 'D':
        return [7]
    elif freq == 'W':
        return [52]
    elif freq in ['M', 'BM', 'CBM', 'SM'] or freq.startswith(('M', 'BM', 'BS', 'CBM', 'SM')):
        return [12]
    elif freq in ['Q', 'BQ', 'REQ'] or freq.startswith(('Q', 'BQ', 'REQ')):
        return [4]
    elif freq in ['H', 'BH', 'CBH']:
        return [24]
    elif freq in ['T', 'min']:
        return [60]
    elif freq == 'S':
        return [60]
    return None

def _compute_samples(model, predictions, n_samples):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function is drawn from Model._calculate_confidence_intervals() in tbats.\n    We have to implement our own version here in order to compute the samples before\n    the inverse boxcox transform.\n    '
    if n_samples == 1:
        return np.expand_dims(predictions, axis=1)
    F = model.matrix.make_F_matrix()
    g = model.matrix.make_g_vector()
    w = model.matrix.make_w_vector()
    c = np.asarray([1.0] * len(predictions))
    f_running = np.identity(F.shape[1])
    for step in range(1, len(predictions)):
        c[step] = w @ f_running @ g
        f_running = f_running @ F
    variance_multiplier = np.cumsum(c * c)
    base_variance_boxcox = np.sum(model.resid_boxcox * model.resid_boxcox) / len(model.y)
    variance_boxcox = base_variance_boxcox * variance_multiplier
    std_boxcox = np.sqrt(variance_boxcox)
    samples = np.random.normal(loc=model._boxcox(predictions), scale=std_boxcox, size=(n_samples, len(predictions))).T
    samples = np.expand_dims(samples, axis=1)
    boxcox_lambda = model.params.box_cox_lambda
    if boxcox_lambda is not None:
        samples = inv_boxcox(samples, boxcox_lambda)
    return samples

class _BaseBatsTbatsModel(LocalForecastingModel, ABC):

    def __init__(self, use_box_cox: Optional[bool]=None, box_cox_bounds: Tuple=(0, 1), use_trend: Optional[bool]=None, use_damped_trend: Optional[bool]=None, seasonal_periods: Optional[Union[str, List]]='freq', use_arma_errors: Optional[bool]=True, show_warnings: bool=False, n_jobs: Optional[int]=None, multiprocessing_start_method: Optional[str]='spawn', random_state: int=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is a wrapper around\n        `tbats\n        <https://github.com/intive-DataScience/tbats>`_.\n\n        This implementation also provides naive frequency inference (when "freq"\n        is provided for ``seasonal_periods``),\n        as well as Darts-compatible sampling of the resulting normal distribution.\n\n        For convenience, the tbats documentation of the parameters is reported here.\n\n        Parameters\n        ----------\n        use_box_cox\n            If Box-Cox transformation of original series should be applied.\n            When ``None`` both cases shall be considered and better is selected by AIC.\n        box_cox_bounds\n            Minimal and maximal Box-Cox parameter values.\n        use_trend\n            Indicates whether to include a trend or not.\n            When ``None``, both cases shall be considered and the better one is selected by AIC.\n        use_damped_trend\n            Indicates whether to include a damping parameter in the trend or not.\n            Applies only when trend is used.\n            When ``None``, both cases shall be considered and the better one is selected by AIC.\n        seasonal_periods\n            Length of each of the periods (amount of observations in each period).\n            TBATS accepts int and float values here.\n            BATS accepts only int values.\n            When ``None`` or empty array, non-seasonal model shall be fitted.\n            If set to ``"freq"``, a single "naive" seasonality\n            based on the series frequency will be used (e.g. [12] for monthly series).\n            In this latter case, the seasonality will be recomputed every time the model is fit.\n        use_arma_errors\n            When True BATS will try to improve the model by modelling residuals with ARMA.\n            Best model will be selected by AIC.\n            If ``False``, ARMA residuals modeling will not be considered.\n        show_warnings\n            If warnings should be shown or not.\n        n_jobs\n            How many jobs to run in parallel when fitting BATS model.\n            When not provided BATS shall try to utilize all available cpu cores.\n        multiprocessing_start_method\n            How threads should be started.\n            See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods\n        random_state\n            Sets the underlying random seed at model initialization time.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import TBATS # or BATS\n        >>> series = AirPassengersDataset().load()\n        >>> # based on preliminary analysis, the series contains a trend\n        >>> model = TBATS(use_trend=True)\n        >>> model.fit(series)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[448.29856017],\n               [439.42215052],\n               [507.73465028],\n               [493.03751671],\n               [498.85885374],\n               [564.64871897]])\n        '
        super().__init__()
        self.kwargs = {'use_box_cox': use_box_cox, 'box_cox_bounds': box_cox_bounds, 'use_trend': use_trend, 'use_damped_trend': use_damped_trend, 'seasonal_periods': seasonal_periods, 'use_arma_errors': use_arma_errors, 'show_warnings': show_warnings, 'n_jobs': n_jobs, 'multiprocessing_start_method': multiprocessing_start_method}
        self.seasonal_periods = seasonal_periods
        self.infer_seasonal_periods = seasonal_periods == 'freq'
        self.model = None
        np.random.seed(random_state)

    @abstractmethod
    def _create_model(self):
        if False:
            return 10
        pass

    def fit(self, series: TimeSeries):
        if False:
            print('Hello World!')
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        if self.infer_seasonal_periods:
            seasonality = _seasonality_from_freq(series)
            self.kwargs['seasonal_periods'] = seasonality
            self.seasonal_periods = seasonality
        model = self._create_model()
        fitted_model = model.fit(series.values())
        self.model = fitted_model
        return self

    def predict(self, n, num_samples=1, verbose: bool=False):
        if False:
            while True:
                i = 10
        super().predict(n, num_samples)
        yhat = self.model.forecast(steps=n)
        samples = _compute_samples(self.model, yhat, num_samples)
        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return False

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return True

    @property
    def min_train_series_length(self) -> int:
        if False:
            print('Hello World!')
        if isinstance(self.seasonal_periods, List) and len(self.seasonal_periods) > 0 and (max(self.seasonal_periods) > 1):
            return 2 * max(self.seasonal_periods)
        return 3

class TBATS(_BaseBatsTbatsModel):

    def _create_model(self):
        if False:
            for i in range(10):
                print('nop')
        return tbats_TBATS(**self.kwargs)

class BATS(_BaseBatsTbatsModel):

    def _create_model(self):
        if False:
            i = 10
            return i + 15
        return tbats_BATS(**self.kwargs)