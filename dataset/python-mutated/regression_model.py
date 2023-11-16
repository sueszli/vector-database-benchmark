"""
Regression Model
----------------
A `RegressionModel` forecasts future values of a target series based on

* The target series (past lags only)

* An optional past_covariates series (past lags only)

* An optional future_covariates series (possibly past and future lags)

* Available static covariates


The regression models are learned in a supervised way, and they can wrap around any "scikit-learn like" regression model
acting on tabular data having ``fit()`` and ``predict()`` methods.

Darts also provides :class:`LinearRegressionModel` and :class:`RandomForest`, which are regression models
wrapping around scikit-learn linear regression and random forest regression, respectively.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.

The lags can be specified either using an integer - in which case it represents the _number_ of (past or future) lags
to take into consideration, or as a list - in which case the lags have to be enumerated (strictly negative values
denoting past lags and positive values including 0 denoting future lags).
When static covariates are present, they are appended to the lagged features. When multiple time series are passed,
if their static covariates do not have the same size, the shorter ones are padded with 0 valued features.
"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.data.tabularization import add_static_covariates_to_lagged_data, create_lagged_component_names, create_lagged_training_data
from darts.utils.historical_forecasts import _check_optimizable_historical_forecasts_global_models, _optimized_historical_forecasts_all_points, _optimized_historical_forecasts_last_points_only, _process_historical_forecast_input
from darts.utils.multioutput import MultiOutputRegressor
from darts.utils.utils import _check_quantiles, get_single_series, seq2series, series2seq
logger = get_logger(__name__)
LAGS_TYPE = Union[int, List[int], Dict[str, Union[int, List[int]]]]
FUTURE_LAGS_TYPE = Union[Tuple[int, int], List[int], Dict[str, Union[Tuple[int, int], List[int]]]]

class RegressionModel(GlobalForecastingModel):

    def __init__(self, lags: Optional[LAGS_TYPE]=None, lags_past_covariates: Optional[LAGS_TYPE]=None, lags_future_covariates: Optional[FUTURE_LAGS_TYPE]=None, output_chunk_length: int=1, add_encoders: Optional[dict]=None, model=None, multi_models: Optional[bool]=True, use_static_covariates: bool=True):
        if False:
            return 10
        "Regression Model\n        Can be used to fit any scikit-learn-like regressor class to predict the target time series from lagged values.\n\n        Parameters\n        ----------\n        lags\n            Lagged target `series` values used to predict the next time step/s.\n            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`\n            corresponds the first predicted time step of each sample.\n            If a list of integers, each value must be < 0. Uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `series` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (integer or list of integers). The\n            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the 'default_lags' key is not provided.\n        lags_past_covariates\n            Lagged `past_covariates` values used to predict the next time step/s.\n            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,\n            where `0` corresponds to the first predicted time step of each sample.\n            If a list of integers, each value must be < 0. Uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (integer or list of integers). The\n            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the 'default_lags' key is not provided.\n        lags_future_covariates\n            Lagged `future_covariates` values used to predict the next time step/s.\n            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`\n            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0`\n            corresponds the first predicted time step of each sample.\n            If a list of integers, uses only the specified values as lags.\n            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when\n            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key\n            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some\n            components are missing and the 'default_lags' key is not provided.\n        output_chunk_length\n            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast\n            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may\n            be useful if the covariates don't extend far enough into the future.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    'cyclic': {'future': ['month']},\n                    'datetime_attribute': {'future': ['hour', 'dayofweek']},\n                    'position': {'past': ['relative'], 'future': ['relative']},\n                    'custom': {'past': [encode_year]},\n                    'transformer': Scaler(),\n                    'tz': 'CET'\n                }\n            ..\n        model\n            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't\n            support multi-output regression for multivariate timeseries, in which case one regressor\n            will be used per component in the multivariate series.\n            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.\n        multi_models\n            If True, a separate model will be trained for each future lag to predict. If False, a single model is\n            trained to predict at step 'output_chunk_length' in the future. Default: True.\n        use_static_covariates\n            Whether the model should use static covariate information in case the input `series` passed to ``fit()``\n            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce\n            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.\n\n        Examples\n        --------\n        >>> from darts.datasets import WeatherDataset\n        >>> from darts.models import RegressionModel\n        >>> from sklearn.linear_model import Ridge\n        >>> series = WeatherDataset().load()\n        >>> # predicting atmospheric pressure\n        >>> target = series['p (mbar)'][:100]\n        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)\n        >>> past_cov = series['rain (mm)'][:100]\n        >>> # optionally, use future temperatures (pretending this component is a forecast)\n        >>> future_cov = series['T (degC)'][:106]\n        >>> # wrap around the sklearn Ridge model\n        >>> model = RegressionModel(\n        >>>     model=Ridge(),\n        >>>     lags=12,\n        >>>     lags_past_covariates=4,\n        >>>     lags_future_covariates=(0,6),\n        >>>     output_chunk_length=6\n        >>> )\n        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[1005.73340676],\n               [1005.71159051],\n               [1005.7322616 ],\n               [1005.76314504],\n               [1005.82204348],\n               [1005.89100967]])\n        "
        super().__init__(add_encoders=add_encoders)
        self.model = model
        self.lags: Dict[str, List[int]] = {}
        self.component_lags: Dict[str, Dict[str, List[int]]] = {}
        self.input_dim = None
        self.multi_models = multi_models
        self._considers_static_covariates = use_static_covariates
        self._static_covariates_shape: Optional[Tuple[int, int]] = None
        self._lagged_feature_names: Optional[List[str]] = None
        raise_if_not(isinstance(output_chunk_length, int) and output_chunk_length > 0, f'output_chunk_length must be an integer greater than 0. Given: {output_chunk_length}', logger=logger)
        self._output_chunk_length = output_chunk_length
        if self.model is None:
            self.model = LinearRegression(n_jobs=-1)
        if not callable(getattr(self.model, 'fit', None)):
            raise_log(Exception('Provided model object must have a fit() method', logger))
        if not callable(getattr(self.model, 'predict', None)):
            raise_log(Exception('Provided model object must have a predict() method', logger))
        raise_if(lags is None and lags_future_covariates is None and (lags_past_covariates is None), 'At least one of `lags`, `lags_future_covariates` or `lags_past_covariates` must be not None.')
        (self.lags, self.component_lags) = self._generate_lags(lags=lags, lags_past_covariates=lags_past_covariates, lags_future_covariates=lags_future_covariates)
        self.pred_dim = self.output_chunk_length if self.multi_models else 1

    def _generate_lags(self, lags: Optional[LAGS_TYPE], lags_past_covariates: Optional[LAGS_TYPE], lags_future_covariates: Optional[FUTURE_LAGS_TYPE]) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, List[int]]]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Based on the type of the argument and the nature of the covariates, perform some sanity checks before\n        converting the lags to a list of integer.\n\n        If lags are provided as a dictionary, the lags values are contained in self.component_lags and the self.lags\n        attributes contain only the extreme values\n        If the lags are provided as integer, list, tuple or dictionary containing only the 'default_lags' keys, the lags\n        values are contained in the self.lags attribute and the self.component_lags is an empty dictionary.\n        "
        processed_lags: Dict[str, List[int]] = dict()
        processed_component_lags: Dict[str, Dict[str, List[int]]] = dict()
        for (lags_values, lags_name, lags_abbrev) in zip([lags, lags_past_covariates, lags_future_covariates], ['lags', 'lags_past_covariates', 'lags_future_covariates'], ['target', 'past', 'future']):
            if lags_values is None:
                continue
            if not isinstance(lags_values, dict):
                lags_values = {'default_lags': lags_values}
            elif len(lags_values) == 0:
                raise_log(ValueError(f'When passed as a dictionary, `{lags_name}` must contain at least one key.'), logger)
            invalid_type = False
            supported_types = ''
            min_lags = None
            max_lags = None
            tmp_components_lags: Dict[str, List[int]] = dict()
            for (comp_name, comp_lags) in lags_values.items():
                if lags_name == 'lags_future_covariates':
                    if isinstance(comp_lags, tuple):
                        raise_if_not(len(comp_lags) == 2 and isinstance(comp_lags[0], int) and isinstance(comp_lags[1], int), f'`{lags_name}` - `{comp_name}`: tuple must be of length 2, and must contain two integers', logger)
                        raise_if(isinstance(comp_lags[0], bool) or isinstance(comp_lags[1], bool), f'`{lags_name}` - `{comp_name}`: tuple must contain integers, not bool', logger)
                        raise_if_not(comp_lags[0] >= 0 and comp_lags[1] >= 0, f'`{lags_name}` - `{comp_name}`: tuple must contain positive integers. Given: {comp_lags}.', logger)
                        raise_if(comp_lags[0] == 0 and comp_lags[1] == 0, f'`{lags_name}` - `{comp_name}`: tuple cannot be (0, 0) as it corresponds to an empty list of lags.', logger)
                        tmp_components_lags[comp_name] = list(range(-comp_lags[0], comp_lags[1]))
                    elif isinstance(comp_lags, list):
                        for lag in comp_lags:
                            raise_if(not isinstance(lag, int) or isinstance(lag, bool), f'`{lags_name}` - `{comp_name}`: list must contain only integers. Given: {comp_lags}.', logger)
                        tmp_components_lags[comp_name] = sorted(comp_lags)
                    else:
                        invalid_type = True
                        supported_types = 'tuple or a list'
                elif isinstance(comp_lags, int):
                    raise_if_not(comp_lags > 0, f'`{lags_name}` - `{comp_name}`: integer must be strictly positive . Given: {comp_lags}.', logger)
                    tmp_components_lags[comp_name] = list(range(-comp_lags, 0))
                elif isinstance(comp_lags, list):
                    for lag in comp_lags:
                        raise_if(not isinstance(lag, int) or lag >= 0, f'`{lags_name}` - `{comp_name}`: list must contain only strictly negative integers. Given: {comp_lags}.', logger)
                    tmp_components_lags[comp_name] = sorted(comp_lags)
                else:
                    invalid_type = True
                    supported_types = 'strictly positive integer or a list'
                if invalid_type:
                    raise_log(ValueError(f'`{lags_name}` - `{comp_name}`: must be either a {supported_types}. Gived : {type(comp_lags)}.'), logger)
                if min_lags is None:
                    min_lags = tmp_components_lags[comp_name][0]
                else:
                    min_lags = min(min_lags, tmp_components_lags[comp_name][0])
                if max_lags is None:
                    max_lags = tmp_components_lags[comp_name][-1]
                else:
                    max_lags = max(max_lags, tmp_components_lags[comp_name][-1])
            if list(tmp_components_lags.keys()) == ['default_lags']:
                processed_lags[lags_abbrev] = tmp_components_lags['default_lags']
            else:
                processed_lags[lags_abbrev] = [min_lags, max_lags]
                processed_component_lags[lags_abbrev] = tmp_components_lags
        return (processed_lags, processed_component_lags)

    def _get_lags(self, lags_type: str):
        if False:
            print('Hello World!')
        '\n        If lags were specified in a component-wise manner, they are contained in self.component_lags and\n        the values in self.lags should be ignored as they correspond just the extreme values.\n        '
        if lags_type in self.component_lags:
            return self.component_lags[lags_type]
        else:
            return self.lags.get(lags_type, None)

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            for i in range(10):
                print('nop')
        target_lags = self.lags.get('target', [0])
        lags_past_covariates = self.lags.get('past', None)
        if lags_past_covariates is not None:
            lags_past_covariates = [min(lags_past_covariates) - int(not self.multi_models) * (self.output_chunk_length - 1), max(lags_past_covariates)]
        lags_future_covariates = self.lags.get('future', None)
        if lags_future_covariates is not None:
            lags_future_covariates = [min(lags_future_covariates) - int(not self.multi_models) * (self.output_chunk_length - 1), max(lags_future_covariates)]
        return (abs(min(target_lags)), self.output_chunk_length, lags_past_covariates is not None, lags_future_covariates is not None, lags_past_covariates, lags_future_covariates)

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            i = 10
            return i + 15
        min_target_lag = self.lags['target'][0] if 'target' in self.lags else None
        max_target_lag = self.output_chunk_length - 1
        min_past_cov_lag = self.lags['past'][0] if 'past' in self.lags else None
        max_past_cov_lag = self.lags['past'][-1] if 'past' in self.lags else None
        min_future_cov_lag = self.lags['future'][0] if 'future' in self.lags else None
        max_future_cov_lag = self.lags['future'][-1] if 'future' in self.lags else None
        return (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag, min_future_cov_lag, max_future_cov_lag)

    @property
    def supports_multivariate(self) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        If available, uses `model`'s native multivariate support. If not available, obtains multivariate support by\n        wrapping the univariate model in a `sklearn.multioutput.MultiOutputRegressor`.\n        "
        return True

    @property
    def min_train_series_length(self) -> int:
        if False:
            return 10
        return max(3, -self.lags['target'][0] + self.output_chunk_length if 'target' in self.lags else self.output_chunk_length)

    @property
    def min_train_samples(self) -> int:
        if False:
            i = 10
            return i + 15
        return 2

    @property
    def output_chunk_length(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._output_chunk_length

    def get_multioutput_estimator(self, horizon, target_dim):
        if False:
            for i in range(10):
                print('nop')
        raise_if_not(isinstance(self.model, MultiOutputRegressor), 'The sklearn model is not a MultiOutputRegressor object.')
        return self.model.estimators_[horizon + target_dim]

    def _create_lagged_data(self, target_series: Sequence[TimeSeries], past_covariates: Sequence[TimeSeries], future_covariates: Sequence[TimeSeries], max_samples_per_ts: int):
        if False:
            i = 10
            return i + 15
        (features, labels, _, self._static_covariates_shape) = create_lagged_training_data(target_series=target_series, output_chunk_length=self.output_chunk_length, past_covariates=past_covariates, future_covariates=future_covariates, lags=self._get_lags('target'), lags_past_covariates=self._get_lags('past'), lags_future_covariates=self._get_lags('future'), uses_static_covariates=self.uses_static_covariates, last_static_covariates_shape=None, max_samples_per_ts=max_samples_per_ts, multi_models=self.multi_models, check_inputs=False, concatenate=False)
        expected_nb_feat = features[0].shape[1] if isinstance(features, Sequence) else features.shape[1]
        for (i, (X_i, y_i)) in enumerate(zip(features, labels)):
            if expected_nb_feat != X_i.shape[1]:
                shape_error_msg = []
                for (ts, cov_name, arg_name) in zip([target_series, past_covariates, future_covariates], ['target', 'past', 'future'], ['series', 'past_covariates', 'future_covariates']):
                    if ts is not None and ts[i].width != self.input_dim[cov_name]:
                        shape_error_msg.append(f'Expected {self.input_dim[cov_name]} components but received {ts[i].width} components at index {i} of `{arg_name}`.')
                raise_log(ValueError('\n'.join(shape_error_msg)), logger)
            features[i] = X_i[:, :, 0]
            labels[i] = y_i[:, :, 0]
        training_samples = np.concatenate(features, axis=0)
        training_labels = np.concatenate(labels, axis=0)
        return (training_samples, training_labels)

    def _fit_model(self, target_series: Sequence[TimeSeries], past_covariates: Sequence[TimeSeries], future_covariates: Sequence[TimeSeries], max_samples_per_ts: int, **kwargs):
        if False:
            print('Hello World!')
        '\n        Function that fit the model. Deriving classes can override this method for adding additional parameters (e.g.,\n        adding validation data), keeping the sanity checks on series performed by fit().\n        '
        (training_samples, training_labels) = self._create_lagged_data(target_series, past_covariates, future_covariates, max_samples_per_ts)
        if len(training_labels.shape) == 2 and training_labels.shape[1] == 1:
            training_labels = training_labels.ravel()
        self.model.fit(training_samples, training_labels, **kwargs)
        (self._lagged_feature_names, _) = create_lagged_component_names(target_series=target_series, past_covariates=past_covariates, future_covariates=future_covariates, lags=self._get_lags('target'), lags_past_covariates=self._get_lags('past'), lags_future_covariates=self._get_lags('future'), output_chunk_length=self.output_chunk_length, concatenate=False, use_static_covariates=self.uses_static_covariates)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, max_samples_per_ts: Optional[int]=None, n_jobs_multioutput_wrapper: Optional[int]=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Fit/train the model on one or multiple series.\n\n        Parameters\n        ----------\n        series\n            TimeSeries or Sequence[TimeSeries] object containing the target values.\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        max_samples_per_ts\n            This is an integer upper bound on the number of tuples that can be produced\n            per time series. It can be used in order to have an upper bound on the total size of the dataset and\n            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset\n            creation) to know their sizes, which might be expensive on big datasets.\n            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the\n            most recent `max_samples_per_ts` samples will be considered.\n        n_jobs_multioutput_wrapper\n            Number of jobs of the MultiOutputRegressor wrapper to run in parallel. Only used if the model doesn't\n            support multi-output regression natively.\n        **kwargs\n            Additional keyword arguments passed to the `fit` method of the model.\n        "
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        self._verify_static_covariates(series[0].static_covariates)
        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            (past_covariates, future_covariates) = self.generate_fit_encodings(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True
        if get_single_series(series).static_covariates is not None and self.supports_static_covariates and self.considers_static_covariates:
            self._uses_static_covariates = True
        for (covs, name) in zip([past_covariates, future_covariates], ['past', 'future']):
            raise_if(covs is not None and name not in self.lags, f'`{name}_covariates` not None in `fit()` method call, but `lags_{name}_covariates` is None in constructor.')
            raise_if(covs is None and name in self.lags, f'`{name}_covariates` is None in `fit()` method call, but `lags_{name}_covariates` is not None in constructor.')
        self.input_dim = {'target': series[0].width, 'past': past_covariates[0].width if past_covariates else None, 'future': future_covariates[0].width if future_covariates else None}
        if not series[0].is_univariate or (self.output_chunk_length > 1 and self.multi_models):
            if not isinstance(self.model, MultiOutputRegressor):
                if not (callable(getattr(self.model, '_get_tags', None)) and isinstance(self.model._get_tags(), dict) and self.model._get_tags().get('multioutput')):
                    self.model = MultiOutputRegressor(self.model, n_jobs=n_jobs_multioutput_wrapper)
                elif self.model.__class__.__name__ == 'CatBoostRegressor':
                    if self.model.get_params()['loss_function'] == 'RMSEWithUncertainty':
                        self.model = MultiOutputRegressor(self.model, n_jobs=n_jobs_multioutput_wrapper)
        if not isinstance(self.model, MultiOutputRegressor) and n_jobs_multioutput_wrapper is not None:
            logger.warning("Provided `n_jobs_multioutput_wrapper` wasn't used.")
        super().fit(series=seq2series(series), past_covariates=seq2series(past_covariates), future_covariates=seq2series(future_covariates))
        variate2arg = {'target': 'lags', 'past': 'lags_past_covariates', 'future': 'lags_future_covariates'}
        component_lags_error_msg = []
        for (variate_type, variate) in zip(['target', 'past', 'future'], [series, past_covariates, future_covariates]):
            if variate_type not in self.component_lags:
                continue
            provided_components = set(self.component_lags[variate_type].keys())
            required_components = set(variate[0].components)
            wrong_components = list(provided_components - {'default_lags'} - required_components)
            missing_keys = list(required_components - provided_components)
            if len(wrong_components) > 0:
                component_lags_error_msg.append(f'The `{variate2arg[variate_type]}` dictionary specifies lags for components that are not present in the series : {wrong_components}. They must be removed to avoid any ambiguity.')
            elif len(missing_keys) > 0 and 'default_lags' not in provided_components:
                component_lags_error_msg.append(f"The {variate2arg[variate_type]} dictionary is missing the lags for the following components present in the series: {missing_keys}. The key 'default_lags' can be used to provide lags for all the non-explicitely defined components.")
            else:
                self.component_lags[variate_type] = {comp_name: self.component_lags[variate_type][comp_name] if comp_name in self.component_lags[variate_type] else self.component_lags[variate_type]['default_lags'] for comp_name in variate[0].components}
        if len(component_lags_error_msg) > 0:
            raise_log(ValueError('\n'.join(component_lags_error_msg)), logger)
        self._fit_model(series, past_covariates, future_covariates, max_samples_per_ts, **kwargs)
        return self

    def predict(self, n: int, series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, num_samples: int=1, verbose: bool=False, predict_likelihood_parameters: bool=False, **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            while True:
                i = 10
        'Forecasts values for `n` time steps after the end of the series.\n\n        Parameters\n        ----------\n        n : int\n            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.\n        series : TimeSeries or list of TimeSeries, optional\n            Optionally, one or several input `TimeSeries`, representing the history of the target series whose future\n            is to be predicted. If specified, the method returns the forecasts of these series. Otherwise, the method\n            returns the forecast of the (single) training series.\n        past_covariates : TimeSeries or list of TimeSeries, optional\n            Optionally, the past-observed covariates series needed as inputs for the model.\n            They must match the covariates used for training in terms of dimension and type.\n        future_covariates : TimeSeries or list of TimeSeries, optional\n            Optionally, the future-known covariates series needed as inputs for the model.\n            They must match the covariates used for training in terms of dimension and type.\n        num_samples : int, default: 1\n            Number of times a prediction is sampled from a probabilistic model. Should be set to 1\n            for deterministic models.\n        verbose\n            Optionally, whether to print progress.\n        predict_likelihood_parameters\n            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only\n            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.\n            Default: ``False``\n        **kwargs : dict, optional\n            Additional keyword arguments passed to the `predict` method of the model. Only works with\n            univariate target series.\n        '
        if series is None:
            if self.training_series is None:
                raise_log(ValueError('Input `series` must be provided. This is the result either from fitting on multiple series, or from not having fit the model yet.'), logger)
            series = self.training_series
        called_with_single_series = True if isinstance(series, TimeSeries) else False
        series = series2seq(series)
        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = [self.past_covariate_series] * len(series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = [self.future_covariate_series] * len(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        self._verify_static_covariates(series[0].static_covariates)
        if self.encoders.encoding_available:
            (past_covariates, future_covariates) = self.generate_predict_encodings(n=n, series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        super().predict(n, series, past_covariates, future_covariates, num_samples, verbose, predict_likelihood_parameters)
        pred_input_dim = {'target': series[0].width, 'past': past_covariates[0].width if past_covariates else None, 'future': future_covariates[0].width if future_covariates else None}
        raise_if_not(pred_input_dim == self.input_dim, f"The number of components of the target series and the covariates provided for prediction doesn't match the number of components of the target series and the covariates this model has been trained on.\nProvided number of components for prediction: {pred_input_dim}\nProvided number of components for training: {self.input_dim}")
        covariates = {'past': (past_covariates, self.lags.get('past')), 'future': (future_covariates, self.lags.get('future'))}
        if self.multi_models:
            shift = 0
            step = self.output_chunk_length
        else:
            shift = self.output_chunk_length - 1
            step = 1
        covariate_matrices = {}
        relative_cov_lags = {}
        for (cov_type, (covs, lags)) in covariates.items():
            if covs is None:
                continue
            relative_cov_lags[cov_type] = np.array(lags) - lags[0]
            covariate_matrices[cov_type] = []
            for (idx, (ts, cov)) in enumerate(zip(series, covs)):
                steps_back = -(min(lags) + 1) + shift
                lags_diff = max(lags) - min(lags) + 1
                n_steps = lags_diff + max(0, n - self.output_chunk_length) + shift
                start_ts = ts.end_time() - ts.freq * steps_back
                end_ts = start_ts + ts.freq * (n_steps - 1)
                if not (cov.start_time() <= start_ts and cov.end_time() >= end_ts):
                    raise_log(ValueError(f"The corresponding {cov_type}_covariate of the series at index {idx} isn't sufficiently long. Given horizon `n={n}`, `min(lags_{cov_type}_covariates)={lags[0]}`, `max(lags_{cov_type}_covariates)={lags[-1]}` and `output_chunk_length={self.output_chunk_length}`, the {cov_type}_covariate has to range from {start_ts} until {end_ts} (inclusive), but it ranges only from {cov.start_time()} until {cov.end_time()}."), logger=logger)
                end_ts = end_ts + ts.freq if ts.has_range_index else end_ts
                covariate_matrices[cov_type].append(cov.slice(start_ts, end_ts).values(copy=False))
            covariate_matrices[cov_type] = np.stack(covariate_matrices[cov_type])
        series_matrix = None
        if 'target' in self.lags:
            series_matrix = np.stack([ts.values(copy=False)[self.lags['target'][0] - shift:, :] for ts in series])
        series_matrix = np.repeat(series_matrix, num_samples, axis=0)
        for (cov_type, data) in covariate_matrices.items():
            covariate_matrices[cov_type] = np.repeat(data, num_samples, axis=0)
        predictions = []
        last_step_shift = 0
        for t_pred in range(0, n, step):
            if 0 < n - t_pred < step and t_pred > 0:
                last_step_shift = t_pred - (n - step)
                t_pred = n - step
            np_X = []
            if 'target' in self.lags:
                if predictions:
                    series_matrix = np.concatenate([series_matrix, predictions[-1]], axis=1)
                if 'target' in self.component_lags:
                    tmp_X = [series_matrix[:, [lag - (shift + last_step_shift) for lag in comp_lags], comp_i] for (comp_i, (comp, comp_lags)) in enumerate(self.component_lags['target'].items())]
                    np_X.append(np.concatenate(tmp_X, axis=1).reshape(len(series) * num_samples, -1))
                else:
                    np_X.append(series_matrix[:, [lag - (shift + last_step_shift) for lag in self.lags['target']]].reshape(len(series) * num_samples, -1))
            for cov_type in ['past', 'future']:
                if cov_type in covariate_matrices:
                    if cov_type in self.component_lags:
                        tmp_X = [covariate_matrices[cov_type][:, np.array(comp_lags) - self.lags[cov_type][0] + t_pred, comp_i] for (comp_i, (comp, comp_lags)) in enumerate(self.component_lags[cov_type].items())]
                        np_X.append(np.concatenate(tmp_X, axis=1).reshape(len(series) * num_samples, -1))
                    else:
                        np_X.append(covariate_matrices[cov_type][:, relative_cov_lags[cov_type] + t_pred].reshape(len(series) * num_samples, -1))
            X = np.concatenate(np_X, axis=1)
            X_blocks = np.split(X, len(series), axis=0)
            (X_blocks, _) = add_static_covariates_to_lagged_data(X_blocks, series, uses_static_covariates=self.uses_static_covariates, last_shape=self._static_covariates_shape)
            X = np.concatenate(X_blocks, axis=0)
            prediction = self._predict_and_sample(X, num_samples, predict_likelihood_parameters, **kwargs)
            predictions.append(prediction[:, last_step_shift:])
        predictions = np.concatenate(predictions, axis=1)[:, :n]
        predictions = np.moveaxis(predictions.reshape(len(series), num_samples, n, -1), 1, -1)
        predictions = [self._build_forecast_series(points_preds=row, input_series=input_tgt, custom_components=self._likelihood_components_names(input_tgt) if predict_likelihood_parameters else None, with_static_covs=False if predict_likelihood_parameters else True, with_hierarchy=False if predict_likelihood_parameters else True) for (idx_ts, (row, input_tgt)) in enumerate(zip(predictions, series))]
        return predictions[0] if called_with_single_series else predictions

    def _predict_and_sample(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        'By default, the regression model returns a single sample.'
        prediction = self.model.predict(x, **kwargs)
        k = x.shape[0]
        return prediction.reshape(k, self.pred_dim, -1)

    @property
    def lagged_feature_names(self) -> Optional[List[str]]:
        if False:
            return 10
        'The lagged feature names the model has been trained on.\n\n        The naming convention for target, past and future covariates is: ``"{name}_{type}_lag{i}"``, where:\n\n            - ``{name}`` the component name of the (first) series\n            - ``{type}`` is the feature type, one of "target", "pastcov", and "futcov"\n            - ``{i}`` is the lag value\n\n        The naming convention for static covariates is: ``"{name}_statcov_target_{comp}"``, where:\n\n            - ``{name}`` the static covariate name of the (first) series\n            - ``{comp}`` the target component name of the (first) that the static covariate act on. If the static\n                covariate acts globally on a multivariate target series, will show "global".\n        '
        return self._lagged_feature_names

    def __str__(self):
        if False:
            print('Hello World!')
        return self.model.__str__()

    @property
    def supports_past_covariates(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self.lags.get('past', [])) > 0

    @property
    def supports_future_covariates(self) -> bool:
        if False:
            print('Hello World!')
        return len(self.lags.get('future', [])) > 0

    @property
    def supports_static_covariates(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def _check_optimizable_historical_forecasts(self, forecast_horizon: int, retrain: Union[bool, int, Callable[..., bool]], show_warnings: bool) -> bool:
        if False:
            return 10
        '\n        Historical forecast can be optimized only if `retrain=False` and `forecast_horizon <= model.output_chunk_length`\n        (no auto-regression required).\n        '
        return _check_optimizable_historical_forecasts_global_models(model=self, forecast_horizon=forecast_horizon, retrain=retrain, show_warnings=show_warnings, allow_autoregression=False)

    def _optimized_historical_forecasts(self, series: Optional[Sequence[TimeSeries]], past_covariates: Optional[Sequence[TimeSeries]]=None, future_covariates: Optional[Sequence[TimeSeries]]=None, num_samples: int=1, start: Optional[Union[pd.Timestamp, float, int]]=None, start_format: Literal['position', 'value']='value', forecast_horizon: int=1, stride: int=1, overlap_end: bool=False, last_points_only: bool=True, verbose: bool=False, show_warnings: bool=True, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]]:
        if False:
            print('Hello World!')
        '\n        For RegressionModels we create the lagged prediction data once per series using a moving window.\n        With this, we can avoid having to recreate the tabular input data and call `model.predict()` for each\n        forecastable index and series.\n        Additionally, there is a dedicated subroutines for `last_points_only=True` and `last_points_only=False`.\n\n        TODO: support forecast_horizon > output_chunk_length (auto-regression)\n        '
        (series, past_covariates, future_covariates) = _process_historical_forecast_input(model=self, series=series, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=forecast_horizon, allow_autoregression=False)
        if last_points_only:
            return _optimized_historical_forecasts_last_points_only(model=self, series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, overlap_end=overlap_end, show_warnings=show_warnings, predict_likelihood_parameters=predict_likelihood_parameters)
        else:
            return _optimized_historical_forecasts_all_points(model=self, series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, overlap_end=overlap_end, show_warnings=show_warnings, predict_likelihood_parameters=predict_likelihood_parameters)

class _LikelihoodMixin:
    """
    A class containing functions supporting quantile, poisson and gaussian regression, to be used as a mixin for some
    `RegressionModel` subclasses.
    """

    @staticmethod
    def _check_likelihood(likelihood, available_likelihoods):
        if False:
            print('Hello World!')
        raise_if_not(likelihood in available_likelihoods, f'If likelihood is specified it must be one of {available_likelihoods}')

    @staticmethod
    def _get_model_container():
        if False:
            return 10
        return _QuantileModelContainer()

    @staticmethod
    def _prepare_quantiles(quantiles):
        if False:
            i = 10
            return i + 15
        if quantiles is None:
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        else:
            quantiles = sorted(quantiles)
            _check_quantiles(quantiles)
        median_idx = quantiles.index(0.5)
        return (quantiles, median_idx)

    def _likelihood_components_names(self, input_series: TimeSeries) -> Optional[List[str]]:
        if False:
            i = 10
            return i + 15
        if self.likelihood == 'quantile':
            return self._quantiles_generate_components_names(input_series)
        elif self.likelihood == 'poisson':
            return self._likelihood_generate_components_names(input_series, ['lambda'])
        else:
            return None

    def _predict_quantile(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        X is of shape (n_series * n_samples, n_regression_features)\n        '
        k = x.shape[0]
        if num_samples == 1 and (not predict_likelihood_parameters):
            fitted = self._model_container[0.5]
            return fitted.predict(x, **kwargs).reshape(k, self.pred_dim, -1)
        model_outputs = []
        for (quantile, fitted) in self._model_container.items():
            self.model = fitted
            model_output = fitted.predict(x, **kwargs).reshape(k, self.pred_dim, -1)
            model_outputs.append(model_output)
        model_outputs = np.stack(model_outputs, axis=-1)
        return model_outputs

    def _predict_poisson(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        X is of shape (n_series * n_samples, n_regression_features)\n        '
        k = x.shape[0]
        return self.model.predict(x, **kwargs).reshape(k, self.pred_dim, -1)

    def _predict_normal(self, x: np.ndarray, num_samples: int, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "Method intended for CatBoost's RMSEWithUncertainty loss. Returns samples\n        computed from double-valued inputs [mean, variance].\n        X is of shape (n_series * n_samples, n_regression_features)\n        "
        k = x.shape[0]
        model_output = self.model.predict(x, **kwargs)
        output_dim = len(model_output.shape)
        if num_samples == 1 and (not predict_likelihood_parameters):
            if output_dim <= 2:
                output_slice = model_output[:, 0]
            else:
                output_slice = model_output[0, :, :]
            return output_slice.reshape(k, self.pred_dim, -1)
        if output_dim <= 2:
            model_output = np.expand_dims(model_output, axis=0)
        else:
            model_output = model_output.transpose()
        return model_output

    def _sampling_quantile(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Sample uniformly between [0, 1] (for each batch example) and return the linear interpolation between the fitted\n        quantiles closest to the sampled value.\n\n        model_output is of shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)\n        '
        (k, n_timesteps, n_components, n_quantiles) = model_output.shape
        probs = self._rng.uniform(size=(k, n_timesteps, n_components, 1))
        probas = np.expand_dims(probs, axis=-2)
        p = np.tile(probas, (1, 1, 1, n_quantiles, 1)).transpose((0, 1, 2, 4, 3))
        tquantiles = np.array(self.quantiles).reshape((1, 1, 1, -1))
        left_idx = np.sum(p > tquantiles, axis=-1)
        right_idx = left_idx + 1
        repeat_count = [1] * n_quantiles
        repeat_count[0] = 2
        repeat_count[-1] = 2
        repeat_count = np.array(repeat_count)
        shifted_output = np.repeat(model_output, repeat_count, axis=-1)
        left_value = np.take_along_axis(shifted_output, left_idx, axis=-1)
        right_value = np.take_along_axis(shifted_output, right_idx, axis=-1)
        ext_quantiles = [0.0] + self.quantiles + [1.0]
        expanded_q = np.tile(np.array(ext_quantiles), left_idx.shape)
        left_q = np.take_along_axis(expanded_q, left_idx, axis=-1)
        right_q = np.take_along_axis(expanded_q, right_idx, axis=-1)
        weights = (probs - left_q) / (right_q - left_q)
        inter = left_value + weights * (right_value - left_value)
        return inter.squeeze(-1)

    def _sampling_poisson(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        model_output is of shape (n_series * n_samples, output_chunk_length, n_components)\n        '
        return self._rng.poisson(lam=model_output).astype(float)

    def _sampling_normal(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        "Sampling method for CatBoost's [mean, variance] output.\n        model_output is of shape (n_components * output_chunk_length, n_samples, 2) where the last dimension\n        contain mu and sigma.\n        "
        (n_entries, n_samples, n_params) = model_output.shape
        mu_sigma_list = [model_output[i, :, :] for i in range(n_entries)]
        list_of_samples = [self._rng.normal(mu_sigma[:, 0], mu_sigma[:, 1]) for mu_sigma in mu_sigma_list]
        samples_transposed = np.array(list_of_samples).transpose()
        samples_reshaped = samples_transposed.reshape(n_samples, self.pred_dim, -1)
        return samples_reshaped

    def _params_quantile(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Quantiles on the last dimension, grouped by component'
        (k, n_timesteps, n_components, n_quantiles) = model_output.shape
        return model_output.reshape(k, n_timesteps, n_components * n_quantiles)

    def _params_poisson(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Lambdas on the last dimension, grouped by component'
        return model_output

    def _params_normal(self, model_output: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '[mu, sigma] on the last dimension, grouped by component'
        shape = model_output.shape
        n_samples = shape[1]
        mu_sigma_list = [model_output[i, :, :] for i in range(shape[0])]
        params_transposed = np.array(mu_sigma_list).transpose()
        params_reshaped = params_transposed.reshape(n_samples, self.pred_dim, -1)
        return params_reshaped

    def _predict_and_sample_likelihood(self, x: np.ndarray, num_samples: int, likelihood: str, predict_likelihood_parameters: bool, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        model_output = getattr(self, f'_predict_{likelihood}')(x, num_samples, predict_likelihood_parameters, **kwargs)
        if predict_likelihood_parameters:
            return getattr(self, f'_params_{likelihood}')(model_output)
        elif num_samples == 1:
            return model_output
        else:
            return getattr(self, f'_sampling_{likelihood}')(model_output)

    def _num_parameters_quantile(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.quantiles)

    def _num_parameters_poisson(self) -> int:
        if False:
            return 10
        return 1

    def _num_parameters_normal(self) -> int:
        if False:
            print('Hello World!')
        return 2

    @property
    def num_parameters(self) -> int:
        if False:
            i = 10
            return i + 15
        'Mimic function of Likelihood class'
        likelihood = self.likelihood
        if likelihood is None:
            return 0
        elif likelihood in ['gaussian', 'RMSEWithUncertainty']:
            return self._num_parameters_normal()
        else:
            return getattr(self, f'_num_parameters_{likelihood}')()

    def _quantiles_generate_components_names(self, input_series: TimeSeries) -> List[str]:
        if False:
            while True:
                i = 10
        return self._likelihood_generate_components_names(input_series, [f'q{quantile:.2f}' for quantile in self._model_container.keys()])

    def _likelihood_generate_components_names(self, input_series: TimeSeries, parameter_names: List[str]) -> List[str]:
        if False:
            print('Hello World!')
        return [f'{tgt_name}_{param_n}' for tgt_name in input_series.components for param_n in parameter_names]

class _QuantileModelContainer(OrderedDict):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

class RegressionModelWithCategoricalCovariates(RegressionModel):

    def __init__(self, lags: Union[int, list]=None, lags_past_covariates: Union[int, List[int]]=None, lags_future_covariates: Union[Tuple[int, int], List[int]]=None, output_chunk_length: int=1, add_encoders: Optional[dict]=None, model=None, multi_models: Optional[bool]=True, use_static_covariates: bool=True, categorical_past_covariates: Optional[Union[str, List[str]]]=None, categorical_future_covariates: Optional[Union[str, List[str]]]=None, categorical_static_covariates: Optional[Union[str, List[str]]]=None):
        if False:
            while True:
                i = 10
        "\n        Extension of `RegressionModel` for regression models that support categorical covariates.\n\n        Parameters\n        ----------\n        lags\n            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags\n            are used (from -1 backward). Otherwise, a list of integers with lags is required (each lag must be < 0).\n        lags_past_covariates\n            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last\n            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers\n            with lags < 0 is required.\n        lags_future_covariates\n            Number of lagged future_covariates values used to predict the next time step. If a tuple (past, future) is\n            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first\n            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list\n            of integers with lags is required.\n        output_chunk_length\n            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast\n            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may\n            be useful if the covariates don't extend far enough into the future.\n        add_encoders\n            A large number of past and future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    'cyclic': {'future': ['month']},\n                    'datetime_attribute': {'future': ['hour', 'dayofweek']},\n                    'position': {'past': ['relative'], 'future': ['relative']},\n                    'custom': {'past': [encode_year]},\n                    'transformer': Scaler(),\n                    'tz': 'CET'\n                }\n            ..\n        model\n            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't\n            support multi-output regression for multivariate timeseries, in which case one regressor\n            will be used per component in the multivariate series.\n            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.\n        multi_models\n            If True, a separate model will be trained for each future lag to predict. If False, a single model is\n            trained to predict at step 'output_chunk_length' in the future. Default: True.\n        use_static_covariates\n            Whether the model should use static covariate information in case the input `series` passed to ``fit()``\n            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce\n            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.\n        categorical_past_covariates\n            Optionally, component name or list of component names specifying the past covariates that should be treated\n            as categorical.\n        categorical_future_covariates\n            Optionally, component name or list of component names specifying the future covariates that should be\n            treated as categorical.\n        categorical_static_covariates\n            Optionally, string or list of strings specifying the static covariates that should be treated as\n            categorical.\n        "
        super().__init__(lags=lags, lags_past_covariates=lags_past_covariates, lags_future_covariates=lags_future_covariates, output_chunk_length=output_chunk_length, add_encoders=add_encoders, model=model, multi_models=multi_models, use_static_covariates=use_static_covariates)
        self.categorical_past_covariates = [categorical_past_covariates] if isinstance(categorical_past_covariates, str) else categorical_past_covariates
        self.categorical_future_covariates = [categorical_future_covariates] if isinstance(categorical_future_covariates, str) else categorical_future_covariates
        self.categorical_static_covariates = [categorical_static_covariates] if isinstance(categorical_static_covariates, str) else categorical_static_covariates

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, max_samples_per_ts: Optional[int]=None, n_jobs_multioutput_wrapper: Optional[int]=None, **kwargs):
        if False:
            return 10
        self._validate_categorical_covariates(series=series, past_covariates=past_covariates, future_covariates=future_covariates)
        super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, n_jobs_multioutput_wrapper=n_jobs_multioutput_wrapper, **kwargs)

    @property
    def _categorical_fit_param(self) -> Tuple[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the name, and default value of the categorical features parameter from model's `fit` method .\n        Can be overridden in subclasses.\n        "
        return ('categorical_feature', 'auto')

    def _validate_categorical_covariates(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Checks that the categorical covariates are valid. Specifically, checks that the categorical covariates\n        of the model are a subset of all covariates.\n\n        Parameters\n        ----------\n        series\n            TimeSeries or Sequence[TimeSeries] object containing the target values.\n        past_covariates\n            Optionally, a series or sequence of series specifying past-observed covariates\n        future_covariates\n            Optionally, a series or sequence of series specifying future-known covariates\n        '
        for (categorical_covariates, covariates, cov_type) in zip([self.categorical_past_covariates, self.categorical_future_covariates], [past_covariates, future_covariates], ['past_covariates', 'future_covariates']):
            if categorical_covariates:
                if not covariates:
                    raise_log(ValueError(f'`categorical_{cov_type}` were declared at model creation but no `{cov_type}` are passed to the `fit()` call.'))
                s = get_single_series(covariates)
                if not set(categorical_covariates).issubset(set(s.components)):
                    raise_log(ValueError(f'Some `categorical_{cov_type}` components ({set(categorical_covariates) - set(s.components)}) declared at model creation are not present in the `{cov_type}` passed to the `fit()` call.'))
        if self.categorical_static_covariates:
            s = get_single_series(series)
            covariates = s.static_covariates
            if not s.has_static_covariates:
                raise_log(ValueError('`categorical_static_covariates` were declared at model creation but `series`passed to the `fit()` call does not contain `static_covariates`.'))
            if not set(self.categorical_static_covariates).issubset(set(covariates.columns)):
                raise_log(ValueError(f"Some `categorical_static_covariates` components ({set(self.categorical_static_covariates) - set(covariates.columns)}) declared at model creation are not present in the series' `static_covariates` passed to the `fit()` call."))

    def _get_categorical_features(self, series: Union[List[TimeSeries], TimeSeries], past_covariates: Optional[Union[List[TimeSeries], TimeSeries]]=None, future_covariates: Optional[Union[List[TimeSeries], TimeSeries]]=None) -> Tuple[List[int], List[str]]:
        if False:
            return 10
        '\n        Returns the indices and column names of the categorical features in the regression model.\n\n        Steps:\n        1. Get the list of features used in the model. We keep the creation order of the different lags/features\n            in create_lagged_data.\n        2. Get the indices of the categorical features in the list of features.\n        '
        categorical_covariates = (self.categorical_past_covariates if self.categorical_past_covariates else []) + (self.categorical_future_covariates if self.categorical_future_covariates else []) + (self.categorical_static_covariates if self.categorical_static_covariates else [])
        if not categorical_covariates:
            return ([], [])
        else:
            target_ts = get_single_series(series)
            past_covs_ts = get_single_series(past_covariates)
            fut_covs_ts = get_single_series(future_covariates)
            feature_list = [f'target_{component}_lag{lag}' for lag in self.lags.get('target', []) for component in target_ts.components] + [f'past_cov_{component}_lag{lag}' for lag in self.lags.get('past', []) for component in past_covs_ts.components] + [f'fut_cov_{component}_lag{lag}' for lag in self.lags.get('future', []) for component in fut_covs_ts.components] + (list(target_ts.static_covariates.columns) if target_ts.has_static_covariates else [])
            indices = [i for (i, col) in enumerate(feature_list) for cat in categorical_covariates if cat and cat in col]
            col_names = [feature_list[i] for i in indices]
            return (indices, col_names)

    def _fit_model(self, target_series, past_covariates, future_covariates, max_samples_per_ts, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Custom fit function for `RegressionModelWithCategoricalCovariates` models, adding logic to let the model\n        handle categorical features directly.\n        '
        (cat_col_indices, _) = self._get_categorical_features(target_series, past_covariates, future_covariates)
        (cat_param_name, cat_param_default) = self._categorical_fit_param
        kwargs[cat_param_name] = cat_col_indices if cat_col_indices else cat_param_default
        super()._fit_model(target_series=target_series, past_covariates=past_covariates, future_covariates=future_covariates, max_samples_per_ts=max_samples_per_ts, **kwargs)