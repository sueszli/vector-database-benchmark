"""
Linear Regression model
-----------------------

A forecasting model using a linear regression of some of the target series' lags, as well as optionally some
covariate series lags in order to obtain a forecast.
"""
from typing import List, Optional, Sequence, Union
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression, PoissonRegressor, QuantileRegressor
from darts.logging import get_logger
from darts.models.forecasting.regression_model import FUTURE_LAGS_TYPE, LAGS_TYPE, RegressionModel, _LikelihoodMixin
from darts.timeseries import TimeSeries
logger = get_logger(__name__)

class LinearRegressionModel(RegressionModel, _LikelihoodMixin):

    def __init__(self, lags: Optional[LAGS_TYPE]=None, lags_past_covariates: Optional[LAGS_TYPE]=None, lags_future_covariates: Optional[FUTURE_LAGS_TYPE]=None, output_chunk_length: int=1, add_encoders: Optional[dict]=None, likelihood: Optional[str]=None, quantiles: Optional[List[float]]=None, random_state: Optional[int]=None, multi_models: Optional[bool]=True, use_static_covariates: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        'Linear regression model.\n\n        Parameters\n        ----------\n        lags\n            Lagged target `series` values used to predict the next time step/s.\n            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`\n            corresponds the first predicted time step of each sample.\n            If a list of integers, each value must be < 0. Uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `series` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (integer or list of integers). The\n            key \'default_lags\' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the \'default_lags\' key is not provided.\n        lags_past_covariates\n            Lagged `past_covariates` values used to predict the next time step/s.\n            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,\n            where `0` corresponds to the first predicted time step of each sample.\n            If a list of integers, each value must be < 0. Uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (integer or list of integers). The\n            key \'default_lags\' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the \'default_lags\' key is not provided.\n        lags_future_covariates\n            Lagged `future_covariates` values used to predict the next time step/s.\n            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`\n            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0`\n            corresponds the first predicted time step of each sample.\n            If a list of integers, uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key\n            \'default_lags\' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the \'default_lags\' key is not provided.\n        output_chunk_length\n            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast\n            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may\n            be useful if the covariates don\'t extend far enough into the future.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'past\': [\'relative\'], \'future\': [\'relative\']},\n                    \'custom\': {\'past\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        likelihood\n            Can be set to `quantile` or `poisson`. If set, the model will be probabilistic, allowing sampling at\n            prediction time. If set to `quantile`, the `sklearn.linear_model.QuantileRegressor` is used. Similarly, if\n            set to `poisson`, the `sklearn.linear_model.PoissonRegressor` is used.\n        quantiles\n            Fit the model to these quantiles if the `likelihood` is set to `quantile`.\n        random_state\n            Control the randomness of the sampling. Used as seed for\n            `numpy.random.Generator\n            <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`_. Ignored when\n            no `likelihood` is set.\n            Default: ``None``.\n        multi_models\n            If True, a separate model will be trained for each future lag to predict. If False, a single model is\n            trained to predict at step \'output_chunk_length\' in the future. Default: True.\n        use_static_covariates\n            Whether the model should use static covariate information in case the input `series` passed to ``fit()``\n            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce\n            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.\n        **kwargs\n            Additional keyword arguments passed to `sklearn.linear_model.LinearRegression` (by default), to\n            `sklearn.linear_model.PoissonRegressor` (if `likelihood="poisson"`), or to\n            `sklearn.linear_model.QuantileRegressor` (if `likelihood="quantile"`).\n\n        Examples\n        --------\n        Deterministic forecasting, using past/future covariates (optional)\n\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import LinearRegressionModel\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series[\'p (mbar)\'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series[\'rain (mm)\'][:100]\n        >>> # optionally, use future temperatures (pretending this component is a forecast)\n        >>> future_cov = series[\'T (degC)\'][:106]\n        >>> # predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature\n        >>> # values corresponding to the forecasted period\n        >>> model = LinearRegressionModel(\n        >>>     lags=12,\n        >>>     lags_past_covariates=12,\n        >>>     lags_future_covariates=[0,1,2,3,4,5],\n        >>>     output_chunk_length=6,\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[1005.72085839],\n               [1005.6548696 ],\n               [1005.65403772],\n               [1005.6846175 ],\n               [1005.75753605],\n               [1005.81830675]])\n        '
        self.kwargs = kwargs
        self._median_idx = None
        self._model_container = None
        self.quantiles = None
        self.likelihood = likelihood
        self._rng = None
        available_likelihoods = ['quantile', 'poisson']
        if likelihood is not None:
            self._check_likelihood(likelihood, available_likelihoods)
            self._rng = np.random.default_rng(seed=random_state)
            if likelihood == 'poisson':
                model = PoissonRegressor(**kwargs)
            if likelihood == 'quantile':
                model = QuantileRegressor(**kwargs)
                (self.quantiles, self._median_idx) = self._prepare_quantiles(quantiles)
                self._model_container = self._get_model_container()
        else:
            model = LinearRegression(**kwargs)
        super().__init__(lags=lags, lags_past_covariates=lags_past_covariates, lags_future_covariates=lags_future_covariates, output_chunk_length=output_chunk_length, add_encoders=add_encoders, model=model, multi_models=multi_models, use_static_covariates=use_static_covariates)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, max_samples_per_ts: Optional[int]=None, n_jobs_multioutput_wrapper: Optional[int]=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Fit/train the model on one or multiple series.\n\n        Parameters\n        ----------\n        series\n            TimeSeries or Sequence[TimeSeries] object containing the target values.\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        max_samples_per_ts\n            This is an integer upper bound on the number of tuples that can be produced\n            per time series. It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        n_jobs_multioutput_wrapper\n            Number of jobs of the MultiOutputRegressor wrapper to run in parallel. Only used if the model doesn't\n            support multi-output regression natively.\n        **kwargs\n            Additional keyword arguments passed to the `fit` method of the model.\n        "
        if self.likelihood == 'quantile':
            if 'solver' not in self.kwargs:
                self.kwargs['solver'] = 'highs'
            c = [1]
            try:
                linprog(c=c, method=self.kwargs['solver'])
            except ValueError as ve:
                logger.warning(f'{ve}. Upgrading scipy enables significantly faster solvers')
                self.kwargs['solver'] = 'interior-point'
            self._model_container.clear()
            for quantile in self.quantiles:
                self.kwargs['quantile'] = quantile
                self.model = QuantileRegressor(**self.kwargs)
                super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, **kwargs)
                self._model_container[quantile] = self.model
            return self
        else:
            super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, **kwargs)
            return self

    def _predict_and_sample(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        if self.likelihood is not None:
            return self._predict_and_sample_likelihood(x, num_samples, self.likelihood, predict_likelihood_parameters, **kwargs)
        else:
            return super()._predict_and_sample(x, num_samples, predict_likelihood_parameters, **kwargs)

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            return 10
        return self.likelihood is not None