"""
CatBoost model
--------------

CatBoost based regression model.

This implementation comes with the ability to produce probabilistic forecasts.
"""
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from catboost import CatBoostRegressor
from darts.logging import get_logger
from darts.models.forecasting.regression_model import RegressionModel, _LikelihoodMixin
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class CatBoostModel(RegressionModel, _LikelihoodMixin):

    def __init__(self, lags: Union[int, list]=None, lags_past_covariates: Union[int, List[int]]=None, lags_future_covariates: Union[Tuple[int, int], List[int]]=None, output_chunk_length: int=1, add_encoders: Optional[dict]=None, likelihood: str=None, quantiles: List=None, random_state: Optional[int]=None, multi_models: Optional[bool]=True, use_static_covariates: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "CatBoost Model\n\n        Parameters\n        ----------\n        lags\n            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags\n            are used (from -1 backward). Otherwise a list of integers with lags is required (each lag must be < 0).\n        lags_past_covariates\n            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last\n            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers\n            with lags < 0 is required.\n        lags_future_covariates\n            Number of lagged future_covariates values used to predict the next time step. If an tuple (past, future) is\n            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first\n            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list\n            of integers with lags is required.\n        output_chunk_length\n            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast\n            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may\n            be useful if the covariates don't extend far enough into the future.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    'cyclic': {'future': ['month']},\n                    'datetime_attribute': {'future': ['hour', 'dayofweek']},\n                    'position': {'past': ['relative'], 'future': ['relative']},\n                    'custom': {'past': [encode_year]},\n                    'transformer': Scaler(),\n                    'tz': 'CET'\n                }\n            ..\n        likelihood\n            Can be set to 'quantile', 'poisson' or 'gaussian'. If set, the model will be probabilistic,\n            allowing sampling at prediction time. When set to 'gaussian', the model will use CatBoost's\n            'RMSEWithUncertainty' loss function. When using this loss function, CatBoost returns a mean\n            and variance couple, which capture data (aleatoric) uncertainty.\n            This will overwrite any `objective` parameter.\n        quantiles\n            Fit the model to these quantiles if the `likelihood` is set to `quantile`.\n        random_state\n            Control the randomness in the fitting procedure and for sampling.\n            Default: ``None``.\n        multi_models\n            If True, a separate model will be trained for each future lag to predict. If False, a single model is\n            trained to predict at step 'output_chunk_length' in the future. Default: True.\n        use_static_covariates\n            Whether the model should use static covariate information in case the input `series` passed to ``fit()``\n            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce\n            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.\n        **kwargs\n            Additional keyword arguments passed to `catboost.CatBoostRegressor`.\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import CatBoostModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series['p (mbar)'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series['rain (mm)'][:100]\n        >>> # optionally, use future temperatures (pretending this component is a forecast)\n        >>> future_cov = series['T (degC)'][:106]\n        >>> # predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature\n        >>> # values corresponding to the forecasted period\n        >>> model = CatBoostModel(\n        >>>     lags=12,\n        >>>     lags_past_covariates=12,\n        >>>     lags_future_covariates=[0,1,2,3,4,5],\n        >>>     output_chunk_length=6\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[1006.4153701 ],\n               [1006.41907237],\n               [1006.30872957],\n               [1006.28614154],\n               [1006.22355514],\n               [1006.21607546]])\n        "
        kwargs['random_state'] = random_state
        self.kwargs = kwargs
        self._median_idx = None
        self._model_container = None
        self._rng = None
        self.likelihood = likelihood
        self.quantiles = None
        self._output_chunk_length = output_chunk_length
        likelihood_map = {'quantile': None, 'poisson': 'Poisson', 'gaussian': 'RMSEWithUncertainty', 'RMSEWithUncertainty': 'RMSEWithUncertainty'}
        available_likelihoods = list(likelihood_map.keys())
        if likelihood is not None:
            self._check_likelihood(likelihood, available_likelihoods)
            self._rng = np.random.default_rng(seed=random_state)
            if likelihood == 'quantile':
                (self.quantiles, self._median_idx) = self._prepare_quantiles(quantiles)
                self._model_container = self._get_model_container()
            else:
                self.kwargs['loss_function'] = likelihood_map[likelihood]
        if 'allow_writing_files' not in kwargs:
            kwargs['allow_writing_files'] = False
        super().__init__(lags=lags, lags_past_covariates=lags_past_covariates, lags_future_covariates=lags_future_covariates, output_chunk_length=output_chunk_length, add_encoders=add_encoders, multi_models=multi_models, model=CatBoostRegressor(**kwargs), use_static_covariates=use_static_covariates)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, max_samples_per_ts: Optional[int]=None, verbose: Optional[Union[int, bool]]=0, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Fits/trains the model using the provided list of features time series and the target time series.\n\n        Parameters\n        ----------\n        series\n            TimeSeries or Sequence[TimeSeries] object containing the target values.\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        val_series\n            TimeSeries or Sequence[TimeSeries] object containing the target values for evaluation dataset\n        val_past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates for evaluation dataset\n        val_future_covariates : Union[TimeSeries, Sequence[TimeSeries]]\n            Optionally, a series or sequence of series specifying future-known covariates for evaluation dataset\n        max_samples_per_ts\n            This is an integer upper bound on the number of tuples that can be produced\n            per time series. It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        verbose\n            An integer or a boolean that can be set to 1 to display catboost's default verbose output\n        **kwargs\n            Additional kwargs passed to `catboost.CatboostRegressor.fit()`\n        "
        if val_series is not None:
            kwargs['eval_set'] = self._create_lagged_data(target_series=val_series, past_covariates=val_past_covariates, future_covariates=val_future_covariates, max_samples_per_ts=max_samples_per_ts)
        if self.likelihood == 'quantile':
            self._model_container.clear()
            for quantile in self.quantiles:
                this_quantile = str(quantile)
                self.kwargs['loss_function'] = f'Quantile:alpha={this_quantile}'
                self.model = CatBoostRegressor(**self.kwargs)
                super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, verbose=verbose, **kwargs)
                self._model_container[quantile] = self.model
            return self
        super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, verbose=verbose, **kwargs)
        return self

    def _predict_and_sample(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            return 10
        "Override of RegressionModel's method to allow for the probabilistic case"
        if self.likelihood in ['gaussian', 'RMSEWithUncertainty']:
            return self._predict_and_sample_likelihood(x, num_samples, 'normal', predict_likelihood_parameters, **kwargs)
        elif self.likelihood is not None:
            return self._predict_and_sample_likelihood(x, num_samples, self.likelihood, predict_likelihood_parameters, **kwargs)
        else:
            return super()._predict_and_sample(x, num_samples, predict_likelihood_parameters, **kwargs)

    def _likelihood_components_names(self, input_series: TimeSeries) -> Optional[List[str]]:
        if False:
            while True:
                i = 10
        "Override of RegressionModel's method to support the gaussian/normal likelihood"
        if self.likelihood == 'quantile':
            return self._quantiles_generate_components_names(input_series)
        elif self.likelihood == 'poisson':
            return self._likelihood_generate_components_names(input_series, ['lamba'])
        elif self.likelihood in ['gaussian', 'RMSEWithUncertainty']:
            return self._likelihood_generate_components_names(input_series, ['mu', 'sigma'])
        else:
            return None

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            print('Hello World!')
        return self.likelihood is not None

    @property
    def min_train_series_length(self) -> int:
        if False:
            i = 10
            return i + 15
        return max(3, -self.lags['target'][0] + self.output_chunk_length + 1 if 'target' in self.lags else self.output_chunk_length)