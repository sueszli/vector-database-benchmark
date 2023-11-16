"""
Forecasting Model Base Classes

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.

The main functions are `fit()` and `predict()`. `fit()` learns the function `f()`, over the history of
one or several time series. The function `predict()` applies `f()` on one or several time series in order
to obtain forecasts for a desired number of time stamps into the future.
"""
import copy
import datetime
import inspect
import io
import os
import pickle
import time
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
from random import sample
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
from darts import metrics
from darts.dataprocessing.encoders import SequentialEncoder
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils import _build_tqdm_iterator, _parallel_apply, _with_sanity_checks
from darts.utils.historical_forecasts.utils import _adjust_historical_forecasts_time_index, _get_historical_forecast_predict_index, _get_historical_forecast_train_index, _historical_forecasts_general_checks, _reconciliate_historical_time_indices
from darts.utils.timeseries_generation import _build_forecast_series, _generate_new_dates, generate_index
from darts.utils.utils import get_single_series, series2seq
logger = get_logger(__name__)

class ModelMeta(ABCMeta):
    """Meta class to store parameters used at model creation.

    When creating a model instance, the parameters are extracted as follows:

        1)  Get the model's __init__ signature and store all arg and kwarg
            names as well as default values (empty for args) in an ordered
            dict `all_params`.
        2)  Replace the arg values from `all_params` with the positional
            args used at model creation.
        3)  Remove args from `all_params` that were not passed as positional
            args at model creation. This will enforce that an error is raised
            if not all positional args were passed. If all positional args
            were passed, no parameter will be removed.
        4)  Update `all_params` kwargs with optional kwargs from model creation.
        5)  Save `all_params` to the model.
        6)  Call (create) the model with `all_params`.
    """

    def __call__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        sig = inspect.signature(cls.__init__)
        all_params = OrderedDict([(p.name, p.default) for p in sig.parameters.values() if not p.name == 'self'])
        for (param, arg) in zip(all_params, args):
            all_params[param] = arg
        remove_params = []
        for (param, val) in all_params.items():
            if val is sig.parameters[param].empty:
                remove_params.append(param)
        for param in remove_params:
            all_params.pop(param)
        all_params.update(kwargs)
        cls._model_call = all_params
        return super().__call__(**all_params)

class ForecastingModel(ABC, metaclass=ModelMeta):
    """The base class for forecasting models. It defines the *minimal* behavior that all forecasting models have to
    support. The signatures in this base class are for "local" models handling only one univariate series and no
    covariates. Sub-classes can handle more complex cases.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.training_series: Optional[TimeSeries] = None
        self.past_covariate_series: Optional[TimeSeries] = None
        self.future_covariate_series: Optional[TimeSeries] = None
        self.static_covariates: Optional[pd.DataFrame] = None
        (self._expect_past_covariates, self._uses_past_covariates) = (False, False)
        (self._expect_future_covariates, self._uses_future_covariates) = (False, False)
        self._considers_static_covariates = False
        (self._expect_static_covariates, self._uses_static_covariates) = (False, False)
        self._fit_called = False
        self._model_params = self._extract_model_creation_params()
        if 'add_encoders' not in kwargs:
            raise_log(NotImplementedError('Model subclass must pass the `add_encoders` parameter to base class.'), logger=logger)
        self.add_encoders = kwargs['add_encoders']
        self.encoders: Optional[SequentialEncoder] = None

    @abstractmethod
    def fit(self, series: TimeSeries) -> 'ForecastingModel':
        if False:
            print('Hello World!')
        'Fit/train the model on the provided series.\n\n        Parameters\n        ----------\n        series\n            A target time series. The model will be trained to forecast this time series.\n\n        Returns\n        -------\n        self\n            Fitted model.\n        '
        raise_if_not(len(series) >= self.min_train_series_length, 'Train series only contains {} elements but {} model requires at least {} entries'.format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True
        if series.has_range_index:
            self._supports_range_index

    @property
    def _supports_range_index(self) -> bool:
        if False:
            while True:
                i = 10
        'Checks if the forecasting model supports a range index.\n        Some models may not support this, if for instance they rely on underlying dates.\n\n        By default, returns True. Needs to be overwritten by models that do not support\n        range indexing and raise meaningful exception.\n        '
        return True

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Checks if the forecasting model supports probabilistic predictions.\n        By default, returns False. Needs to be overwritten by models that do support\n        probabilistic predictions.\n        '
        return False

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the forecasting model supports historical forecasts without retraining\n        the model. By default, returns False. Needs to be overwritten by models that do\n        support historical forecasts without retraining.\n        '
        return False

    @property
    @abstractmethod
    def supports_multivariate(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether the model considers more than one variate in the time series.\n        '

    @property
    def supports_past_covariates(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether model supports past covariates\n        '
        return 'past_covariates' in inspect.signature(self.fit).parameters.keys()

    @property
    def supports_future_covariates(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether model supports future covariates\n        '
        return 'future_covariates' in inspect.signature(self.fit).parameters.keys()

    @property
    def supports_static_covariates(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether model supports static covariates\n        '
        return False

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        if False:
            return 10
        '\n        Whether model instance supports direct prediction of likelihood parameters\n        '
        return getattr(self, 'likelihood', None) is not None

    @property
    def uses_past_covariates(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether the model uses past covariates, once fitted.\n        '
        return self._uses_past_covariates

    @property
    def uses_future_covariates(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether the model uses future covariates, once fitted.\n        '
        return self._uses_future_covariates

    @property
    def uses_static_covariates(self) -> bool:
        if False:
            return 10
        '\n        Whether the model uses static covariates, once fitted.\n        '
        return self._uses_static_covariates

    @property
    def considers_static_covariates(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether the model considers static covariates, if there are any.\n        '
        return self._considers_static_covariates

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether the model supports optimized historical forecasts\n        '
        return False

    @property
    def output_chunk_length(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        '\n        Number of time steps predicted at once by the model, not defined for statistical models.\n        '
        return None

    @abstractmethod
    def predict(self, n: int, num_samples: int=1) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        'Forecasts values for `n` time steps after the end of the training series.\n\n        Parameters\n        ----------\n        n\n            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n\n        Returns\n        -------\n        TimeSeries\n            A time series containing the `n` next points after then end of the training series.\n        '
        if not self._fit_called:
            raise_log(ValueError('The model must be fit before calling predict(). For global models, if predict() is called without specifying a series, the model must have been fit on a single training series.'), logger)
        if not self._is_probabilistic and num_samples > 1:
            raise_log(ValueError('`num_samples > 1` is only supported for probabilistic models.'), logger)

    def _fit_wrapper(self, series: TimeSeries, past_covariates: Optional[TimeSeries], future_covariates: Optional[TimeSeries]):
        if False:
            return 10
        self.fit(series)

    def _predict_wrapper(self, n: int, series: TimeSeries, past_covariates: Optional[TimeSeries], future_covariates: Optional[TimeSeries], num_samples: int, verbose: bool=False, predict_likelihood_parameters: bool=False) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        kwargs = dict()
        if self.supports_likelihood_parameter_prediction:
            kwargs['predict_likelihood_parameters'] = predict_likelihood_parameters
        return self.predict(n, num_samples=num_samples, verbose=verbose, **kwargs)

    @property
    def min_train_series_length(self) -> int:
        if False:
            return 10
        '\n        The minimum required length for the training series.\n        '
        return 3

    @property
    def min_train_samples(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        The minimum number of samples for training the model.\n        '
        return 1

    @property
    @abstractmethod
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        A 6-tuple containing in order:\n        (min target lag, max target lag, min past covariate lag, max past covariate lag, min future covariate\n        lag, max future covariate lag). If 0 is the index of the first prediction, then all lags are relative to this\n        index.\n\n        See examples below.\n\n        If the model wasn't fitted with:\n            - target (concerning RegressionModels only): then the first element should be `None`.\n\n            - past covariates: then the third and fourth elements should be `None`.\n\n            - future covariates: then the fifth and sixth elements should be `None`.\n\n        Should be overridden by models that use past or future covariates, and/or for model that have minimum target\n        lag and maximum target lags potentially different from -1 and 0.\n\n        Notes\n        -----\n        maximum target lag (second value) cannot be `None` and is always larger than or equal to 0.\n        Examples\n        --------\n        >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)\n        >>> model.fit(train_series)\n        >>> model.extreme_lags\n        (-3, 1, None, None, None, None)\n        >>> model = LinearRegressionModel(lags=[-3, -5], lags_past_covariates = 4, output_chunk_length=7)\n        >>> model.fit(train_series, past_covariates=past_covariates)\n        >>> model.extreme_lags\n        (-5, 6, -4, -1,  None, None)\n        >>> model = LinearRegressionModel(lags=[3, 5], lags_future_covariates = [4, 6], output_chunk_length=7)\n        >>> model.fit(train_series, future_covariates=future_covariates)\n        >>> model.extreme_lags\n        (-5, 6, None, None, 4, 6)\n        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7)\n        >>> model.fit(train_series)\n        >>> model.extreme_lags\n        (-10, 6, None, None, None, None)\n        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7, lags_future_covariates=[4, 6])\n        >>> model.fit(train_series, future_covariates)\n        >>> model.extreme_lags\n        (-10, 6, None, None, 4, 6)\n        "
        pass

    @property
    def _training_sample_time_index_length(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Required time_index length for one training sample, for any model.\n        '
        (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag, min_future_cov_lag, max_future_cov_lag) = self.extreme_lags
        return max(max_target_lag + 1, max_future_cov_lag + 1 if max_future_cov_lag else 0) - min(min_target_lag if min_target_lag else 0, min_past_cov_lag if min_past_cov_lag else 0, min_future_cov_lag if min_future_cov_lag else 0)

    @property
    def _predict_sample_time_index_length(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Required time_index length for one `predict` function call, for any model.\n        '
        (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag, min_future_cov_lag, max_future_cov_lag) = self.extreme_lags
        return (max_future_cov_lag + 1 if max_future_cov_lag else 0) - min(min_target_lag if min_target_lag else 0, min_past_cov_lag if min_past_cov_lag else 0, min_future_cov_lag if min_future_cov_lag else 0)

    @property
    def _predict_sample_time_index_past_length(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Required time_index length in the past for one `predict` function call, for any model.\n        '
        (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag, min_future_cov_lag, max_future_cov_lag) = self.extreme_lags
        return -min(min_target_lag if min_target_lag else 0, min_past_cov_lag if min_past_cov_lag else 0, min_future_cov_lag if min_future_cov_lag else 0)

    def _generate_new_dates(self, n: int, input_series: Optional[TimeSeries]=None) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        if False:
            return 10
        '\n        Generates `n` new dates after the end of the specified series\n        '
        input_series = input_series if input_series is not None else self.training_series
        return _generate_new_dates(n=n, input_series=input_series)

    def _build_forecast_series(self, points_preds: Union[np.ndarray, Sequence[np.ndarray]], input_series: Optional[TimeSeries]=None, custom_components: Union[List[str], None]=None, with_static_covs: bool=True, with_hierarchy: bool=True, pred_start: Optional[Union[pd.Timestamp, int]]=None) -> TimeSeries:
        if False:
            return 10
        '\n        Builds a forecast time series starting after the end of the training time series, with the\n        correct time index (or after the end of the input series, if specified).\n\n        Parameters\n        ----------\n        points_preds\n            Forecasted values, can be either the target(s) or parameters of the likelihood model\n        input_series\n            TimeSeries used as input for the prediction\n        custom_components\n            New names for the forecast TimeSeries components, used when the number of components changes\n        with_static_covs\n            If set to False, do not copy the input_series `static_covariates` attribute\n        with_hierarchy\n            If set to False, do not copy the input_series `hierarchy` attribute\n        pred_start\n            Optionally, give a custom prediction start point.\n\n        Returns\n        -------\n        TimeSeries\n            New TimeSeries instance starting after the input series\n        '
        input_series = input_series if input_series is not None else self.training_series
        return _build_forecast_series(points_preds, input_series, custom_components, with_static_covs, with_hierarchy, pred_start)

    def _historical_forecasts_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Sanity checks for the historical_forecasts function\n\n        Parameters\n        ----------\n        args\n            The args parameter(s) provided to the historical_forecasts function.\n        kwargs\n            The kwargs parameter(s) provided to the historical_forecasts function.\n\n        Raises\n        ------\n        ValueError\n            when a check on the parameter does not pass.\n        '
        series = args[0]
        _historical_forecasts_general_checks(self, series, kwargs)

    def _get_last_prediction_time(self, series, forecast_horizon, overlap_end, latest_possible_prediction_start):
        if False:
            while True:
                i = 10
        if overlap_end:
            return latest_possible_prediction_start
        return series.time_index[-forecast_horizon]

    def _check_optimizable_historical_forecasts(self, forecast_horizon: int, retrain: Union[bool, int, Callable[..., bool]], show_warnings: bool) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'By default, historical forecasts cannot be optimized'
        return False

    @_with_sanity_checks('_historical_forecasts_sanity_checks')
    def historical_forecasts(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, num_samples: int=1, train_length: Optional[int]=None, start: Optional[Union[pd.Timestamp, float, int]]=None, start_format: Literal['position', 'value']='value', forecast_horizon: int=1, stride: int=1, retrain: Union[bool, int, Callable[..., bool]]=True, overlap_end: bool=False, last_points_only: bool=True, verbose: bool=False, show_warnings: bool=True, predict_likelihood_parameters: bool=False, enable_optimization: bool=True) -> Union[TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]]:
        if False:
            for i in range(10):
                print('nop')
        'Compute the historical forecasts that would have been obtained by this model on\n        (potentially multiple) `series`.\n\n        This method repeatedly builds a training set: either expanding from the beginning of `series` or moving with\n        a fixed length `train_length`. It trains the model on the training set, emits a forecast of length equal to\n        forecast_horizon, and then moves the end of the training set forward by `stride` time steps.\n\n        By default, this method will return one (or a sequence of) single time series made up of\n        the last point of each historical forecast.\n        This time series will thus have a frequency of ``series.freq * stride``.\n        If `last_points_only` is set to False, it will instead return one (or a sequence of) list of the\n        historical forecasts series.\n\n        By default, this method always re-trains the models on the entire available history, corresponding to an\n        expanding window strategy. If `retrain` is set to False, the model must have been fit before. This is not\n        supported by all models.\n\n        Parameters\n        ----------\n        series\n            The (or a sequence of) target time series used to successively train and compute the historical forecasts.\n        past_covariates\n            Optionally, one (or a sequence of) past-observed covariate series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            Optionally, one (or a sequence of) of future-known covariate series. This applies only if the model\n            supports future covariates.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Use values `>1` only for probabilistic\n            models.\n        train_length\n            Number of time steps in our training set (size of backtesting window to train on). Only effective when\n            `retrain` is not ``False``. Default is set to `train_length=None` where it takes all available time steps\n            up until prediction time, otherwise the moving window strategy is used. If larger than the number of time\n            steps available, all steps up until prediction time are used, as in default case. Needs to be at least\n            `min_train_series_length`.\n        start\n            Optionally, the first point in time at which a prediction is computed. This parameter supports:\n            ``float``, ``int``, ``pandas.Timestamp``, and ``None``.\n            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.\n            If an ``int``, it is either the index position of the first prediction point for `series` with a\n            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to\n            the index position with `start_format="position"`.\n            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.\n            If ``None``, the first prediction point will automatically be set to:\n\n            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first\n              predictable point is earlier than the first trainable point.\n            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),\n              or `retrain` is a Callable and the first trainable point is earlier than the first predictable point.\n            - the first trainable point (given `train_length`) otherwise\n\n            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.\n            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter\n            (default behavior with ``None``) and start at the first trainable/predictable point.\n        start_format\n            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a\n            `pd.RangeIndex`.\n            If set to \'position\', `start` corresponds to the index position of the first predicted point and can range\n            from `(-len(series), len(series) - 1)`.\n            If set to \'value\', `start` corresponds to the index value/label of the first predicted point. Will raise\n            an error if the value is not in `series`\' index. Default: ``\'value\'``\n        forecast_horizon\n            The forecast horizon for the predictions.\n        stride\n            The number of time steps between two consecutive predictions.\n        retrain\n            Whether and/or on which condition to retrain the model before predicting.\n            This parameter supports 3 different datatypes: ``bool``, (positive) ``int``, and\n            ``Callable`` (returning a ``bool``).\n            In the case of ``bool``: retrain the model at each step (`True`), or never retrains the model (`False`).\n            In the case of ``int``: the model is retrained every `retrain` iterations.\n            In the case of ``Callable``: the model is retrained whenever callable returns `True`.\n            The callable must have the following positional arguments:\n\n            - `counter` (int): current `retrain` iteration\n            - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)\n            - `train_series` (TimeSeries): train series up to `pred_time`\n            - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`\n            - `future_covariates` (TimeSeries): future_covariates series up\n              to `min(pred_time + series.freq * forecast_horizon, series.end_time())`\n\n            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed\n            to the corresponding retrain function argument.\n            Note: some models do require being retrained every time and do not support anything other\n            than `retrain=True`.\n        overlap_end\n            Whether the returned forecasts can go beyond the series\' end or not.\n        last_points_only\n            Whether to retain only the last point of each historical forecast.\n            If set to True, the method returns a single ``TimeSeries`` containing the successive point forecasts.\n            Otherwise, returns a list of historical ``TimeSeries`` forecasts.\n        verbose\n            Whether to print progress.\n        show_warnings\n            Whether to show warnings related to historical forecasts optimization, or parameters `start` and\n            `train_length`.\n        predict_likelihood_parameters\n            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only\n            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.\n            Default: ``False``\n        enable_optimization\n            Whether to use the optimized version of historical_forecasts when supported and available.\n\n        Returns\n        -------\n        TimeSeries or List[TimeSeries] or List[List[TimeSeries]]\n            If `last_points_only` is set to True and a single series is provided in input, a single ``TimeSeries``\n            is returned, which contains the historical forecast at the desired horizon.\n\n            A ``List[TimeSeries]`` is returned if either `series` is a ``Sequence`` of ``TimeSeries``,\n            or if `last_points_only` is set to False. A list of lists is returned if both conditions are met.\n            In this last case, the outer list is over the series provided in the input sequence,\n            and the inner lists contain the different historical forecasts.\n        '
        model: ForecastingModel = self
        base_class_name = model.__class__.__base__.__name__
        raise_if(not model._fit_called and retrain is False, 'The model has not been fitted yet, and `retrain` is ``False``. Either call `fit()` before `historical_forecasts()`, or set `retrain` to something different than ``False``.', logger)
        raise_if((isinstance(retrain, Callable) or int(retrain) != 1) and (not model._supports_non_retrainable_historical_forecasts), f'{base_class_name} does not support historical forecasting with `retrain` set to `False`. For now, this is only supported with GlobalForecastingModels such as TorchForecastingModels. For more information, read the documentation for `retrain` in `historical_forecasts()`', logger)
        if train_length and (not isinstance(train_length, int)):
            raise_log(TypeError('If not None, train_length needs to be an integer.'), logger)
        elif train_length is not None and train_length < 1:
            raise_log(ValueError('If not None, train_length needs to be positive.'), logger)
        elif train_length is not None and train_length < model._training_sample_time_index_length + (model.min_train_samples - 1):
            raise_log(ValueError(f'train_length is too small for the training requirements of this model. Must be `>={model._training_sample_time_index_length + (model.min_train_samples - 1)}`.'), logger)
        if train_length is not None and retrain is False:
            raise_log(ValueError('cannot use `train_length` when `retrain=False`.'), logger)
        if isinstance(retrain, bool) or (isinstance(retrain, int) and retrain >= 0):

            def retrain_func(counter, pred_time, train_series, past_covariates, future_covariates):
                if False:
                    for i in range(10):
                        print('nop')
                return counter % int(retrain) == 0 if retrain else False
        elif isinstance(retrain, Callable):
            retrain_func = retrain
            expected_arguments = ['counter', 'pred_time', 'train_series', 'past_covariates', 'future_covariates']
            passed_arguments = list(inspect.signature(retrain_func).parameters.keys())
            raise_if(expected_arguments != passed_arguments, f'the Callable `retrain` must have a signature/arguments matching the following positional arguments: {expected_arguments}.', logger)
            result = retrain_func(counter=0, pred_time=get_single_series(series).time_index[-1], train_series=series, past_covariates=past_covariates, future_covariates=future_covariates)
            raise_if_not(isinstance(result, bool), f'Return value of `retrain` must be bool, received {type(result)}', logger)
        else:
            retrain_func = None
            raise_log(ValueError('`retrain` argument must be either `bool`, positive `int` or `Callable` (returning `bool`)'), logger)
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        if enable_optimization and model.supports_optimized_historical_forecasts and model._check_optimizable_historical_forecasts(forecast_horizon=forecast_horizon, retrain=retrain, show_warnings=show_warnings):
            return model._optimized_historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, overlap_end=overlap_end, last_points_only=last_points_only, verbose=verbose, show_warnings=show_warnings, predict_likelihood_parameters=predict_likelihood_parameters)
        if len(series) == 1:
            outer_iterator = series
        else:
            outer_iterator = _build_tqdm_iterator(series, verbose)
        forecasts_list = []
        for (idx, series_) in enumerate(outer_iterator):
            past_covariates_ = past_covariates[idx] if past_covariates else None
            future_covariates_ = future_covariates[idx] if future_covariates else None
            historical_forecasts_time_index_predict = _get_historical_forecast_predict_index(model, series_, idx, past_covariates_, future_covariates_, forecast_horizon, overlap_end)
            if retrain:
                historical_forecasts_time_index_train = _get_historical_forecast_train_index(model, series_, idx, past_covariates_, future_covariates_, forecast_horizon, overlap_end)
                min_timestamp_series = historical_forecasts_time_index_train[0] - model._training_sample_time_index_length * series_.freq
                (historical_forecasts_time_index, train_length_) = _reconciliate_historical_time_indices(model=model, historical_forecasts_time_index_predict=historical_forecasts_time_index_predict, historical_forecasts_time_index_train=historical_forecasts_time_index_train, series=series_, series_idx=idx, retrain=retrain, train_length=train_length, show_warnings=show_warnings)
            else:
                min_timestamp_series = series_.time_index[0]
                historical_forecasts_time_index = historical_forecasts_time_index_predict
                train_length_ = None
            historical_forecasts_time_index = _adjust_historical_forecasts_time_index(series=series_, series_idx=idx, historical_forecasts_time_index=historical_forecasts_time_index, start=start, start_format=start_format, show_warnings=show_warnings)
            if min_timestamp_series > series_.time_index[0]:
                series_ = series_.drop_before(min_timestamp_series - 1 * series_.freq)
            historical_forecasts_time_index = generate_index(start=historical_forecasts_time_index[0], end=historical_forecasts_time_index[-1], freq=series_.freq)
            if len(series) == 1:
                iterator = _build_tqdm_iterator(historical_forecasts_time_index[::stride], verbose)
            else:
                iterator = historical_forecasts_time_index[::stride]
            forecasts = []
            last_points_times = []
            last_points_values = []
            _counter_train = 0
            forecast_components = None
            for (_counter, pred_time) in enumerate(iterator):
                if pred_time <= series_.end_time():
                    train_series = series_.drop_after(pred_time)
                else:
                    train_series = series_
                if train_length_ and len(train_series) > train_length_:
                    train_series = train_series[-train_length_:]
                if retrain and historical_forecasts_time_index_train is not None and (historical_forecasts_time_index_train[0] <= pred_time <= historical_forecasts_time_index_train[-1]):
                    if retrain_func(counter=_counter_train, pred_time=pred_time, train_series=train_series, past_covariates=past_covariates_, future_covariates=future_covariates_):
                        model = model.untrained_model()
                        model._fit_wrapper(series=train_series, past_covariates=past_covariates_, future_covariates=future_covariates_)
                    elif not _counter_train and (not model._fit_called):
                        raise_log(ValueError(f'`retrain` is `False` in the first train iteration at prediction point (in time) `{pred_time}` and the model has not been fit before. Either call `fit()` before `historical_forecasts()`, use a different `retrain` value or modify the function to return `True` at or before this timestamp.'), logger)
                    _counter_train += 1
                elif not _counter and (not model._fit_called):
                    raise_log(ValueError(f"Model has not been fit before the first predict iteration at prediction point (in time) `{pred_time}`. Either call `fit()` before `historical_forecasts()`, set `retrain=True`, modify the function to return `True` at least once before `{pred_time}`, or use a different `start` value. The first 'predictable' timestamp with re-training inside `historical_forecasts` is: {historical_forecasts_time_index_train[0]} (potential `start` value)."), logger)
                if len(train_series) == 0:
                    train_series = TimeSeries.from_times_and_values(times=generate_index(start=pred_time - 1 * series_.freq, length=1, freq=series_.freq), values=np.array([np.NaN]))
                forecast = model._predict_wrapper(n=forecast_horizon, series=train_series, past_covariates=past_covariates_, future_covariates=future_covariates_, num_samples=num_samples, verbose=verbose, predict_likelihood_parameters=predict_likelihood_parameters)
                if forecast_components is None:
                    forecast_components = forecast.columns
                if last_points_only:
                    last_points_values.append(forecast.all_values(copy=False)[-1])
                    last_points_times.append(forecast.end_time())
                else:
                    forecasts.append(forecast)
            if last_points_only and last_points_values:
                forecasts_list.append(TimeSeries.from_times_and_values(generate_index(start=last_points_times[0], end=last_points_times[-1], freq=series_.freq * stride), np.array(last_points_values), columns=forecast_components if forecast_components is not None else series_.columns, static_covariates=series_.static_covariates if not predict_likelihood_parameters else None, hierarchy=series_.hierarchy if not predict_likelihood_parameters else None))
            else:
                forecasts_list.append(forecasts)
        return forecasts_list if len(series) > 1 else forecasts_list[0]

    def backtest(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, historical_forecasts: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, num_samples: int=1, train_length: Optional[int]=None, start: Optional[Union[pd.Timestamp, float, int]]=None, start_format: Literal['position', 'value']='value', forecast_horizon: int=1, stride: int=1, retrain: Union[bool, int, Callable[..., bool]]=True, overlap_end: bool=False, last_points_only: bool=False, metric: Union[Callable[[TimeSeries, TimeSeries], float], List[Callable[[TimeSeries, TimeSeries], float]]]=metrics.mape, reduction: Union[Callable[[np.ndarray], float], None]=np.mean, verbose: bool=False, show_warnings: bool=True) -> Union[float, List[float], Sequence[float], List[Sequence[float]]]:
        if False:
            return 10
        'Compute error values that the model would have produced when\n        used on (potentially multiple) `series`.\n\n        If `historical_forecasts` are provided, the metric (given by the `metric` function) is evaluated directly on\n        the forecast and the actual values. Otherwise, it repeatedly builds a training set: either expanding from the\n        beginning  of `series` or moving with a fixed length `train_length`. It trains the current model on the\n        training set, emits a forecast of length equal to `forecast_horizon`, and then moves the end of the training\n        set forward by `stride` time steps. The metric is then evaluated on the forecast and the actual values.\n        Finally, the method returns a `reduction` (the mean by default) of all these metric scores.\n\n        By default, this method uses each historical forecast (whole) to compute error scores.\n        If `last_points_only` is set to True, it will use only the last point of each historical\n        forecast. In this case, no reduction is used.\n\n        By default, this method always re-trains the models on the entire available history, corresponding to an\n        expanding window strategy. If `retrain` is set to False (useful for models for which training might be\n        time-consuming, such as deep learning models), the trained model will be used directly to emit the forecasts.\n\n        Parameters\n        ----------\n        series\n            The (or a sequence of) target time series used to successively train and evaluate the historical forecasts.\n        past_covariates\n            Optionally, one (or a sequence of) past-observed covariate series. This applies only if the model\n            supports past covariates.\n        future_covariates\n            Optionally, one (or a sequence of) future-known covariate series. This applies only if the model\n            supports future covariates.\n        historical_forecasts\n            Optionally, the (or a sequence of) historical forecasts time series to be evaluated. Corresponds to\n            the output of :meth:`historical_forecasts() <ForecastingModel.historical_forecasts>`. If provided, will\n            skip historical forecasting and ignore all parameters except `series`, `metric`, and `reduction`.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Use values `>1` only for probabilistic\n            models.\n        train_length\n            Number of time steps in our training set (size of backtesting window to train on). Only effective when\n            `retrain` is not ``False``. Default is set to `train_length=None` where it takes all available time steps\n            up until prediction time, otherwise the moving window strategy is used. If larger than the number of time\n            steps available, all steps up until prediction time are used, as in default case. Needs to be at least\n            `min_train_series_length`.\n        start\n            Optionally, the first point in time at which a prediction is computed. This parameter supports:\n            ``float``, ``int``, ``pandas.Timestamp``, and ``None``.\n            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.\n            If an ``int``, it is either the index position of the first prediction point for `series` with a\n            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to\n            the index position with `start_format="position"`.\n            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.\n            If ``None``, the first prediction point will automatically be set to:\n\n            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first\n              predictable point is earlier than the first trainable point.\n            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),\n              or `retrain` is a Callable and the first trainable point is earlier than the first predictable point.\n            - the first trainable point (given `train_length`) otherwise\n\n            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.\n            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter\n            (default behavior with ``None``) and start at the first trainable/predictable point.\n        start_format\n            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a\n            `pd.RangeIndex`.\n            If set to \'position\', `start` corresponds to the index position of the first predicted point and can range\n            from `(-len(series), len(series) - 1)`.\n            If set to \'value\', `start` corresponds to the index value/label of the first predicted point. Will raise\n            an error if the value is not in `series`\' index. Default: ``\'value\'``\n        forecast_horizon\n            The forecast horizon for the point predictions.\n        stride\n            The number of time steps between two consecutive predictions.\n        retrain\n            Whether and/or on which condition to retrain the model before predicting.\n            This parameter supports 3 different datatypes: ``bool``, (positive) ``int``, and\n            ``Callable`` (returning a ``bool``).\n            In the case of ``bool``: retrain the model at each step (`True`), or never retrains the model (`False`).\n            In the case of ``int``: the model is retrained every `retrain` iterations.\n            In the case of ``Callable``: the model is retrained whenever callable returns `True`.\n            The callable must have the following positional arguments:\n\n                - `counter` (int): current `retrain` iteration\n                - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)\n                - `train_series` (TimeSeries): train series up to `pred_time`\n                - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`\n                - `future_covariates` (TimeSeries): future_covariates series up\n                  to `min(pred_time + series.freq * forecast_horizon, series.end_time())`\n\n            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed\n            to the corresponding retrain function argument.\n            Note: some models do require being retrained every time and do not support anything other\n            than `retrain=True`.\n        overlap_end\n            Whether the returned forecasts can go beyond the series\' end or not.\n        last_points_only\n            Whether to use the whole historical forecasts or only the last point of each forecast to compute the error.\n        metric\n            A function or a list of function that takes two ``TimeSeries`` instances as inputs and returns an\n            error value.\n        reduction\n            A function used to combine the individual error scores obtained when `last_points_only` is set to False.\n            When providing several metric functions, the function will receive the argument `axis = 0` to obtain single\n            value for each metric function.\n            If explicitly set to `None`, the method will return a list of the individual error scores instead.\n            Set to ``np.mean`` by default.\n        verbose\n            Whether to print progress.\n        show_warnings\n            Whether to show warnings related to parameters `start`, and `train_length`.\n\n        Returns\n        -------\n        float or List[float] or List[List[float]]\n            The (sequence of) error score on a series, or list of list containing error scores for each\n            provided series and each sample.\n        '
        if historical_forecasts is None:
            forecasts = self.historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, train_length=train_length, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, retrain=retrain, overlap_end=overlap_end, last_points_only=last_points_only, verbose=verbose, show_warnings=show_warnings)
        else:
            forecasts = historical_forecasts
        series = series2seq(series)
        if len(series) == 1:
            forecasts = [forecasts]
        if not isinstance(metric, list):
            metric = [metric]
        backtest_list = []
        for (idx, target_ts) in enumerate(series):
            if last_points_only:
                errors = [metric_f(target_ts, forecasts[idx]) for metric_f in metric]
                if len(errors) == 1:
                    errors = errors[0]
                backtest_list.append(errors)
            else:
                errors = [[metric_f(target_ts, f) for metric_f in metric] if len(metric) > 1 else metric[0](target_ts, f) for f in forecasts[idx]]
                if reduction is None:
                    backtest_list.append(errors)
                else:
                    backtest_list.append(reduction(np.array(errors), axis=0))
        return backtest_list if len(backtest_list) > 1 else backtest_list[0]

    @classmethod
    def gridsearch(model_class, parameters: dict, series: TimeSeries, past_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, forecast_horizon: Optional[int]=None, stride: int=1, start: Union[pd.Timestamp, float, int]=0.5, start_format: Literal['position', 'value']='value', last_points_only: bool=False, show_warnings: bool=True, val_series: Optional[TimeSeries]=None, use_fitted_values: bool=False, metric: Callable[[TimeSeries, TimeSeries], float]=metrics.mape, reduction: Callable[[np.ndarray], float]=np.mean, verbose=False, n_jobs: int=1, n_random_samples: Optional[Union[int, float]]=None) -> Tuple['ForecastingModel', Dict[str, Any], float]:
        if False:
            return 10
        '\n        Find the best hyper-parameters among a given set using a grid search.\n\n        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.\n        The three modes of operation evaluate every possible combination of hyper-parameter values\n        provided in the `parameters` dictionary by instantiating the `model_class` subclass\n        of ForecastingModel with each combination, and returning the best-performing model with regard\n        to the `metric` function. The `metric` function is expected to return an error value,\n        thus the model resulting in the smallest `metric` output will be chosen.\n\n        The relationship of the training data and test data depends on the mode of operation.\n\n        Expanding window mode (activated when `forecast_horizon` is passed):\n        For every hyperparameter combination, the model is repeatedly trained and evaluated on different\n        splits of `series`. This process is accomplished by using\n        the :func:`backtest` function as a subroutine to produce historic forecasts starting from `start`\n        that are compared against the ground truth values of `series`.\n        Note that the model is retrained for every single prediction, thus this mode is slower.\n\n        Split window mode (activated when `val_series` is passed):\n        This mode will be used when the `val_series` argument is passed.\n        For every hyper-parameter combination, the model is trained on `series` and\n        evaluated on `val_series`.\n\n        Fitted value mode (activated when `use_fitted_values` is set to `True`):\n        For every hyper-parameter combination, the model is trained on `series`\n        and evaluated on the resulting fitted values.\n        Not all models have fitted values, and this method raises an error if the model doesn\'t have a `fitted_values`\n        member. The fitted values are the result of the fit of the model on `series`. Comparing with the\n        fitted values can be a quick way to assess the model, but one cannot see if the model is overfitting the series.\n\n        Derived classes must ensure that a single instance of a model will not share parameters with the other\n        instances, e.g., saving models in the same path. Otherwise, an unexpected behavior can arise while running\n        several models in parallel (when ``n_jobs != 1``). If this cannot be avoided, then gridsearch\n        should be redefined, forcing ``n_jobs = 1``.\n\n        Currently this method only supports deterministic predictions (i.e. when models\' predictions\n        have only 1 sample).\n\n        Parameters\n        ----------\n        model_class\n            The ForecastingModel subclass to be tuned for \'series\'.\n        parameters\n            A dictionary containing as keys hyperparameter names, and as values lists of values for the\n            respective hyperparameter.\n        series\n            The target series used as input and target for training.\n        past_covariates\n            Optionally, a past-observed covariate series. This applies only if the model supports past covariates.\n        future_covariates\n            Optionally, a future-known covariate series. This applies only if the model supports future covariates.\n        forecast_horizon\n            The integer value of the forecasting horizon. Activates expanding window mode.\n        stride\n            Only used in expanding window mode. The number of time steps between two consecutive predictions.\n        start\n            Only used in expanding window mode. Optionally, the first point in time at which a prediction is computed.\n            This parameter supports: ``float``, ``int``, ``pandas.Timestamp``, and ``None``.\n            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.\n            If an ``int``, it is either the index position of the first prediction point for `series` with a\n            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to\n            the index position with `start_format="position"`.\n            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.\n            If ``None``, the first prediction point will automatically be set to:\n\n            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first\n              predictable point is earlier than the first trainable point.\n            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),\n              or `retrain` is a Callable and the first trainable point is earlier than the first predictable point.\n            - the first trainable point (given `train_length`) otherwise\n\n            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.\n            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter\n            (default behavior with ``None``) and start at the first trainable/predictable point.\n        start_format\n            Only used in expanding window mode. Defines the `start` format. Only effective when `start` is an integer\n            and `series` is indexed with a `pd.RangeIndex`.\n            If set to \'position\', `start` corresponds to the index position of the first predicted point and can range\n            from `(-len(series), len(series) - 1)`.\n            If set to \'value\', `start` corresponds to the index value/label of the first predicted point. Will raise\n            an error if the value is not in `series`\' index. Default: ``\'value\'``\n        last_points_only\n            Only used in expanding window mode. Whether to use the whole forecasts or only the last point of each\n            forecast to compute the error.\n        show_warnings\n            Only used in expanding window mode. Whether to show warnings related to the `start` parameter.\n        val_series\n            The TimeSeries instance used for validation in split mode. If provided, this series must start right after\n            the end of `series`; so that a proper comparison of the forecast can be made.\n        use_fitted_values\n            If `True`, uses the comparison with the fitted values.\n            Raises an error if ``fitted_values`` is not an attribute of `model_class`.\n        metric\n            A function that takes two TimeSeries instances as inputs (actual and prediction, in this order),\n            and returns a float error value.\n        reduction\n            A reduction function (mapping array to float) describing how to aggregate the errors obtained\n            on the different validation series when backtesting. By default it\'ll compute the mean of errors.\n        verbose\n            Whether to print progress.\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when there are two or more parameters\n            combinations to evaluate. Each job will instantiate, train, and evaluate a different instance of the model.\n            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available cores.\n        n_random_samples\n            The number/ratio of hyperparameter combinations to select from the full parameter grid. This will perform\n            a random search instead of using the full grid.\n            If an integer, `n_random_samples` is the number of parameter combinations selected from the full grid and\n            must be between `0` and the total number of parameter combinations.\n            If a float, `n_random_samples` is the ratio of parameter combinations selected from the full grid and must\n            be between `0` and `1`. Defaults to `None`, for which random selection will be ignored.\n\n        Returns\n        -------\n        ForecastingModel, Dict, float\n            A tuple containing an untrained `model_class` instance created from the best-performing hyper-parameters,\n            along with a dictionary containing these best hyper-parameters,\n            and metric score for the best hyper-parameters.\n        '
        raise_if_not((forecast_horizon is not None) + (val_series is not None) + use_fitted_values == 1, "Please pass exactly one of the arguments 'forecast_horizon', 'val_target_series' or 'use_fitted_values'.", logger)
        if use_fitted_values:
            raise_if_not(hasattr(model_class(), 'fitted_values'), 'The model must have a fitted_values attribute to compare with the train TimeSeries', logger)
        elif val_series is not None:
            raise_if_not(series.width == val_series.width, 'Training and validation series require the same number of components.', logger)
        params_cross_product = list(product(*parameters.values()))
        if n_random_samples is not None:
            params_cross_product = model_class._sample_params(params_cross_product, n_random_samples)
        iterator = _build_tqdm_iterator(zip(params_cross_product), verbose, total=len(params_cross_product))

        def _evaluate_combination(param_combination) -> float:
            if False:
                print('Hello World!')
            param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
            if param_combination_dict.get('model_name', None):
                current_time = time.strftime('%Y-%m-%d_%H.%M.%S.%f', time.localtime())
                param_combination_dict['model_name'] = f"{current_time}_{param_combination_dict['model_name']}"
            model = model_class(**param_combination_dict)
            if use_fitted_values:
                model._fit_wrapper(series, past_covariates, future_covariates)
                fitted_values = TimeSeries.from_times_and_values(series.time_index, model.fitted_values)
                error = metric(series, fitted_values)
            elif val_series is None:
                error = model.backtest(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=1, start=start, start_format=start_format, forecast_horizon=forecast_horizon, stride=stride, metric=metric, reduction=reduction, last_points_only=last_points_only, verbose=verbose, show_warnings=show_warnings)
            else:
                model._fit_wrapper(series, past_covariates, future_covariates)
                pred = model._predict_wrapper(len(val_series), series, past_covariates, future_covariates, num_samples=1, verbose=verbose)
                error = metric(val_series, pred)
            return float(error)
        errors: List[float] = _parallel_apply(iterator, _evaluate_combination, n_jobs, {}, {})
        min_error = min(errors)
        best_param_combination = dict(list(zip(parameters.keys(), params_cross_product[errors.index(min_error)])))
        logger.info('Chosen parameters: ' + str(best_param_combination))
        return (model_class(**best_param_combination), best_param_combination, min_error)

    def residuals(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, forecast_horizon: int=1, retrain: bool=True, verbose: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            while True:
                i = 10
        'Compute the residuals produced by this model on a (or sequence of) univariate  time series.\n\n        This function computes the difference between the actual observations from `series` and the fitted values\n        vector `p` obtained by training the model on `series`. For every index `i` in `series`, `p[i]` is computed\n        by training the model on ``series[:(i - forecast_horizon)]`` and forecasting `forecast_horizon` into the future.\n        (`p[i]` will be set to the last value of the predicted series.)\n        The vector of residuals will be shorter than `series` due to the minimum training series length required by the\n        model and the gap introduced by `forecast_horizon`. Most commonly, the term "residuals" implies a value for\n        `forecast_horizon` of 1; but this can be configured.\n\n        This method works only on univariate series. It uses the median\n        prediction (when dealing with stochastic forecasts).\n\n        Parameters\n        ----------\n        series\n            The univariate TimeSeries instance which the residuals will be computed for.\n        past_covariates\n            One or several past-observed covariate time series.\n        future_covariates\n            One or several future-known covariate time series.\n        forecast_horizon\n            The forecasting horizon used to predict each fitted value.\n        retrain\n            Whether to train the model at each iteration, for models that support it.\n            If False, the model is not trained at all. Default: True\n        verbose\n            Whether to print progress.\n\n        Returns\n        -------\n        TimeSeries (or Sequence[TimeSeries])\n            The vector of residuals.\n        '
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        raise_if_not(all([serie.is_univariate for serie in series]), 'Each series in the sequence must be univariate.', logger)
        residuals_list = []
        for (idx, target_ts) in enumerate(series):
            first_index = target_ts.time_index[self.min_train_series_length]
            forecasts = self.historical_forecasts(series=target_ts, past_covariates=past_covariates[idx] if past_covariates else None, future_covariates=future_covariates[idx] if future_covariates else None, start=first_index, forecast_horizon=forecast_horizon, stride=1, retrain=retrain, last_points_only=True, verbose=verbose)
            series_trimmed = target_ts.slice_intersect(forecasts)
            residuals_list.append(series_trimmed - (forecasts.quantile_timeseries(quantile=0.5) if forecasts.is_stochastic else forecasts))
        return residuals_list if len(residuals_list) > 1 else residuals_list[0]

    def initialize_encoders(self) -> SequentialEncoder:
        if False:
            while True:
                i = 10
        'instantiates the SequentialEncoder object based on self._model_encoder_settings and parameter\n        ``add_encoders`` used at model creation'
        (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates, lags_past_covariates, lags_future_covariates) = self._model_encoder_settings
        return SequentialEncoder(add_encoders=self.add_encoders, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, lags_past_covariates=lags_past_covariates, lags_future_covariates=lags_future_covariates, takes_past_covariates=takes_past_covariates, takes_future_covariates=takes_future_covariates)

    def generate_fit_encodings(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]:
        if False:
            while True:
                i = 10
        'Generates the covariate encodings that were used/generated for fitting the model and returns a tuple of\n        past, and future covariates series with the original and encoded covariates stacked together. The encodings are\n        generated by the encoders defined at model creation with parameter `add_encoders`. Pass the same `series`,\n        `past_covariates`, and  `future_covariates` that you used to train/fit the model.\n\n        Parameters\n        ----------\n        series\n            The series or sequence of series with the target values used when fitting the model.\n        past_covariates\n            Optionally, the series or sequence of series with the past-observed covariates used when fitting the model.\n        future_covariates\n            Optionally, the series or sequence of series with the future-known covariates used when fitting the model.\n\n        Returns\n        -------\n        Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]\n            A tuple of (past covariates, future covariates). Each covariate contains the original as well as the\n            encoded covariates.\n        '
        raise_if(self.encoders is None or not self.encoders.encoding_available, 'Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the model with `model.fit()` before.', logger=logger)
        return self.encoders.encode_train(target=series, past_covariates=past_covariates, future_covariates=future_covariates)

    def generate_predict_encodings(self, n: int, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]:
        if False:
            return 10
        'Generates covariate encodings for the inference/prediction set and returns a tuple of past, and future\n        covariates series with the original and encoded covariates stacked together. The encodings are generated by the\n        encoders defined at model creation with parameter `add_encoders`. Pass the same `series`, `past_covariates`,\n        and `future_covariates` that you intend to use for prediction.\n\n        Parameters\n        ----------\n        n\n            The number of prediction time steps after the end of `series` intended to be used for prediction.\n        series\n            The series or sequence of series with target values intended to be used for prediction.\n        past_covariates\n            Optionally, the past-observed covariates series intended to be used for prediction. The dimensions must\n            match those of the covariates used for training.\n        future_covariates\n            Optionally, the future-known covariates series intended to be used for prediction. The dimensions must\n            match those of the covariates used for training.\n\n        Returns\n        -------\n        Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]\n            A tuple of (past covariates, future covariates). Each covariate contains the original as well as the\n            encoded covariates.\n        '
        raise_if(self.encoders is None or not self.encoders.encoding_available, 'Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the model with `model.fit()` before.', logger=logger)
        return self.encoders.encode_inference(n=n, target=series, past_covariates=past_covariates, future_covariates=future_covariates)

    def generate_fit_predict_encodings(self, n: int, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]:
        if False:
            while True:
                i = 10
        'Generates covariate encodings for training and inference/prediction and returns a tuple of past, and future\n        covariates series with the original and encoded covariates stacked together. The encodings are generated by the\n        encoders defined at model creation with parameter `add_encoders`. Pass the same `series`, `past_covariates`,\n        and `future_covariates` that you intend to use for training and prediction.\n\n        Parameters\n        ----------\n        n\n            The number of prediction time steps after the end of `series` intended to be used for prediction.\n        series\n            The series or sequence of series with target values intended to be used for training and prediction.\n        past_covariates\n            Optionally, the past-observed covariates series intended to be used for training and prediction. The\n            dimensions must match those of the covariates used for training.\n        future_covariates\n            Optionally, the future-known covariates series intended to be used for prediction. The dimensions must\n            match those of the covariates used for training.\n\n        Returns\n        -------\n        Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]\n            A tuple of (past covariates, future covariates). Each covariate contains the original as well as the\n            encoded covariates.\n        '
        raise_if(self.encoders is None or not self.encoders.encoding_available, 'Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the model with `model.fit()` before.', logger=logger)
        return self.encoders.encode_train_inference(n=n, target=series, past_covariates=past_covariates, future_covariates=future_covariates)

    @property
    @abstractmethod
    def _model_encoder_settings(self) -> Tuple[Optional[int], Optional[int], bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            return 10
        'Abstract property that returns model specific encoder settings that are used to initialize the encoders.\n\n        Must return Tuple (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates,\n        lags_past_covariates, lags_future_covariates).\n        '
        pass

    @classmethod
    def _sample_params(model_class, params, n_random_samples):
        if False:
            for i in range(10):
                print('nop')
        'Select the absolute number of samples randomly if an integer has been supplied. If a float has been\n        supplied, select a fraction'
        if isinstance(n_random_samples, int):
            raise_if_not(n_random_samples > 0 and n_random_samples <= len(params), 'If supplied as an integer, n_random_samples must be greater than 0 and lessthan or equal to the size of the cartesian product of the hyperparameters.')
            return sample(params, n_random_samples)
        if isinstance(n_random_samples, float):
            raise_if_not(n_random_samples > 0.0 and n_random_samples <= 1.0, 'If supplied as a float, n_random_samples must be greater than 0.0 and less than 1.0.')
            return sample(params, int(n_random_samples * len(params)))

    def _extract_model_creation_params(self):
        if False:
            i = 10
            return i + 15
        'extracts immutable model creation parameters from `ModelMeta` and deletes reference.'
        model_params = copy.deepcopy(self._model_call)
        del self.__class__._model_call
        return model_params

    def untrained_model(self):
        if False:
            while True:
                i = 10
        'Returns a new (untrained) model instance create with the same parameters.'
        return self.__class__(**copy.deepcopy(self.model_params))

    @property
    def model_params(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return self._model_params if hasattr(self, '_model_params') else self._model_call

    @classmethod
    def _default_save_path(cls) -> str:
        if False:
            return 10
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"

    def save(self, path: Optional[Union[str, os.PathLike, BinaryIO]]=None, **pkl_kwargs) -> None:
        if False:
            return 10
        '\n        Saves the model under a given path or file handle.\n\n        Example for saving and loading a :class:`RegressionModel`:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                from darts.models import RegressionModel\n\n                model = RegressionModel(lags=4)\n\n                model.save("my_model.pkl")\n                model_loaded = RegressionModel.load("my_model.pkl")\n            ..\n\n        Parameters\n        ----------\n        path\n            Path or file handle under which to save the model at its current state. If no path is specified, the model\n            is automatically saved under ``"{ModelClass}_{YYYY-mm-dd_HH_MM_SS}.pkl"``.\n            E.g., ``"RegressionModel_2020-01-01_12_00_00.pkl"``.\n        pkl_kwargs\n            Keyword arguments passed to `pickle.dump()`\n        '
        if path is None:
            path = self._default_save_path() + '.pkl'
        if isinstance(path, (str, os.PathLike)):
            with open(path, 'wb') as handle:
                pickle.dump(obj=self, file=handle, **pkl_kwargs)
        elif isinstance(path, io.BufferedWriter):
            pickle.dump(obj=self, file=path, **pkl_kwargs)
        else:
            raise_log(ValueError(f"Argument 'path' has to be either 'str' or 'PathLike' (for a filepath) or 'BufferedWriter' (for an already opened file), but was '{path.__class__}'."), logger=logger)

    @staticmethod
    def load(path: Union[str, os.PathLike, BinaryIO]) -> 'ForecastingModel':
        if False:
            while True:
                i = 10
        '\n        Loads the model from a given path or file handle.\n\n        Parameters\n        ----------\n        path\n            Path or file handle from which to load the model.\n        '
        if isinstance(path, (str, os.PathLike)):
            raise_if_not(os.path.exists(path), f"The file {path} doesn't exist", logger)
            with open(path, 'rb') as handle:
                model = pickle.load(file=handle)
        elif isinstance(path, io.BufferedReader):
            model = pickle.load(file=path)
        else:
            raise_log(ValueError(f"Argument 'path' has to be either 'str' or 'PathLike' (for a filepath) or 'BufferedReader' (for an already opened file), but was '{path.__class__}'."), logger=logger)
        return model

    def _assert_univariate(self, series: TimeSeries):
        if False:
            for i in range(10):
                print('nop')
        if not series.is_univariate:
            raise_log(ValueError(f'Model `{self.__class__.__name__}` only supports univariate TimeSeries instances'), logger=logger)

    def _assert_multivariate(self, series: TimeSeries):
        if False:
            return 10
        if series.is_univariate:
            raise_log(ValueError(f'Model `{self.__class__.__name__}` only supports multivariate TimeSeries instances'), logger=logger)

    def __repr__(self):
        if False:
            return 10
        '\n        Get full description for this estimator (includes all params).\n        '
        return self._get_model_description_string(True)

    def __str__(self):
        if False:
            print('Hello World!')
        '\n        Get short description for this estimator (only includes params with non-default values).\n        '
        return self._get_model_description_string(False)

    def _get_model_description_string(self, include_default_params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get model description string of structure `model_name`(`model_param_key_value_pairs`).\n\n        Parameters\n        ----------\n        include_default_params : bool,\n            If True, will include params with default values in the description.\n\n        Returns\n        -------\n        description : String\n            Model description.\n        '
        default_model_params = self._get_default_model_params()
        changed_model_params = [(k, v) for (k, v) in self.model_params.items() if include_default_params or np.any(v != default_model_params.get(k, None))]
        model_name = self.__class__.__name__
        params_string = ', '.join([f'{k}={str(v)}' for (k, v) in changed_model_params])
        return f'{model_name}({params_string})'

    @classmethod
    def _get_default_model_params(cls):
        if False:
            i = 10
            return i + 15
        'Get parameter key : default_value pairs for the estimator'
        init_signature = inspect.signature(cls.__init__)
        return {p.name: p.default for p in init_signature.parameters.values() if p.name != 'self'}

    def _verify_static_covariates(self, static_covariates: Optional[pd.DataFrame]):
        if False:
            i = 10
            return i + 15
        '\n        Verify that all static covariates are numeric.\n        '
        if static_covariates is not None and self.uses_static_covariates:
            numeric_mask = static_covariates.columns.isin(static_covariates.select_dtypes(include=np.number))
            if sum(~numeric_mask):
                raise_log(ValueError(f'{self.__class__.__name__} can only interpret numeric static covariate data. Consider encoding/transforming categorical static covariates with `darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer` or set `use_static_covariates=False` at model creation to ignore static covariates.'), logger)

    def _optimized_historical_forecasts(self, series: Optional[Sequence[TimeSeries]], past_covariates: Optional[Sequence[TimeSeries]]=None, future_covariates: Optional[Sequence[TimeSeries]]=None, num_samples: int=1, start: Optional[Union[pd.Timestamp, float, int]]=None, start_format: Literal['position', 'value']='value', forecast_horizon: int=1, stride: int=1, overlap_end: bool=False, last_points_only: bool=True, verbose: bool=False, show_warnings: bool=True, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]]:
        if False:
            return 10
        logger.warning('`optimized historical forecasts is not available for this model, use `historical_forecasts` instead.')
        return []

class LocalForecastingModel(ForecastingModel, ABC):
    """The base class for "local" forecasting models, handling only single univariate time series.

    Local Forecasting Models (LFM) are models that can be trained on a single univariate target series only. In Darts,
    most models in this category tend to be simpler statistical models (such as ETS or FFT). LFMs usually train on
    the entire target series supplied when calling :func:`fit()` at once. They can also predict in one go with
    :func:`predict()` for any number of predictions `n` after the end of the training series.

    All implementations must implement the `fit()` and `predict()` methods.
    """

    def __init__(self, add_encoders: Optional[dict]=None):
        if False:
            while True:
                i = 10
        super().__init__(add_encoders=add_encoders)

    @property
    def _model_encoder_settings(self) -> Tuple[Optional[int], Optional[int], bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            for i in range(10):
                print('nop')
        return (None, None, False, False, None, None)

    @abstractmethod
    def fit(self, series: TimeSeries) -> 'LocalForecastingModel':
        if False:
            i = 10
            return i + 15
        super().fit(series)
        series._assert_deterministic()

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            print('Hello World!')
        return (-self.min_train_series_length, -1, None, None, None, None)

class GlobalForecastingModel(ForecastingModel, ABC):
    """The base class for "global" forecasting models, handling several time series and optional covariates.

    Global forecasting models expand upon the functionality of `ForecastingModel` in 4 ways:
    1. Models can be fitted on many series (multivariate or univariate) with different indices.
    2. The input series used by :func:`predict()` can be different from the series used to fit the model.
    3. Covariates can be supported (multivariate or univariate).
    4. They can allow for multivariate target series and covariates.

    The name "global" stems from the fact that a training set of a forecasting model of this class is not constrained
    to a temporally contiguous, "local", time series.

    All implementations must implement the :func:`fit()` and :func:`predict()` methods.
    The :func:`fit()` method is meant to train the model on one or several training time series, along with optional
    covariates.

    If :func:`fit()` has been called with only one training and covariate series as argument, then
    calling :func:`predict()` will forecast the future of this series. Otherwise, the user has to
    provide to :func:`predict()` the series they want to forecast, as well as covariates, if needed.
    """

    def __init__(self, add_encoders: Optional[dict]=None):
        if False:
            print('Hello World!')
        super().__init__(add_encoders=add_encoders)

    @abstractmethod
    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> 'GlobalForecastingModel':
        if False:
            while True:
                i = 10
        'Fit/train the model on (potentially multiple) series.\n\n        Optionally, one or multiple past and/or future covariates series can be provided as well.\n        The number of covariates series must match the number of target series.\n\n        Parameters\n        ----------\n        series\n            One or several target time series. The model will be trained to forecast these time series.\n            The series may or may not be multivariate, but if multiple series are provided\n            they must have the same number of components.\n        past_covariates\n            One or several past-observed covariate time series. These time series will not be forecast, but can\n            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple\n            covariates are provided they must have the same number of components. If `past_covariates` is provided,\n            it must contain the same number of series as `series`.\n        future_covariates\n            One or several future-known covariate time series. These time series will not be forecast, but can\n            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple\n            covariates are provided they must have the same number of components. If `future_covariates` is provided,\n            it must contain the same number of series as `series`.\n\n        Returns\n        -------\n        self\n            Fitted model.\n        '
        if isinstance(series, TimeSeries):
            self.training_series = series
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
            if series.static_covariates is not None and self.supports_static_covariates and self.considers_static_covariates:
                self.static_covariates = series.static_covariates
        else:
            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True
            if get_single_series(series).static_covariates is not None and self.supports_static_covariates and self.considers_static_covariates:
                self.static_covariates = series[0].static_covariates
                self._expect_static_covariates = True
        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True
        if get_single_series(series).static_covariates is not None and self.supports_static_covariates and self.considers_static_covariates:
            self._uses_static_covariates = True
        self._fit_called = True

    @abstractmethod
    def predict(self, n: int, series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, num_samples: int=1, verbose: bool=False, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            return 10
        "Forecasts values for `n` time steps after the end of the series.\n\n        If :func:`fit()` has been called with only one ``TimeSeries`` as argument, then the `series` argument of\n        this function is optional, and it will simply produce the next `horizon` time steps forecast.\n        The `past_covariates` and `future_covariates` arguments also don't have to be provided again in this case.\n\n        If :func:`fit()` has been called with `series` specified as a ``Sequence[TimeSeries]`` (i.e., the model\n        has been trained on multiple time series), the `series` argument must be specified.\n\n        When the `series` argument is specified, this function will compute the next `n` time steps forecasts\n        for the simple series (or for each series in the sequence) given by `series`.\n\n        If multiple past or future covariates were specified during the training, some corresponding covariates must\n        also be specified here. For every input in `series` a matching (past and/or future) covariate time series\n        has to be provided.\n\n        Parameters\n        ----------\n        n\n            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.\n        series\n            The series whose future values will be predicted.\n        past_covariates\n            One past-observed covariate time series for every input time series in `series`. They must match the\n            past covariates that have been used with the :func:`fit()` function for training in terms of dimension.\n        future_covariates\n            One future-known covariate time series for every input time series in `series`. They must match the\n            past covariates that have been used with the :func:`fit()` function for training in terms of dimension.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n        verbose\n            Optionally, whether to print progress.\n        predict_likelihood_parameters\n            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only\n            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.\n            Default: ``False``\n\n        Returns\n        -------\n        Union[TimeSeries, Sequence[TimeSeries]]\n            If `series` is not specified, this function returns a single time series containing the `n`\n            next points after then end of the training series.\n            If `series` is given and is a simple ``TimeSeries``, this function returns the `n` next points\n            after the end of `series`.\n            If `series` is given and is a sequence of several time series, this function returns\n            a sequence where each element contains the corresponding `n` points forecasts.\n        "
        super().predict(n, num_samples)
        if predict_likelihood_parameters:
            self._sanity_check_predict_likelihood_parameters(n, self.output_chunk_length, num_samples)
        if self.uses_past_covariates and past_covariates is None:
            raise_log(ValueError('The model has been trained with past covariates. Some matching past_covariates have to be provided to `predict()`.'))
        if self.uses_future_covariates and future_covariates is None:
            raise_log(ValueError('The model has been trained with future covariates. Some matching future_covariates have to be provided to `predict()`.'))
        if self.uses_static_covariates and get_single_series(series).static_covariates is None:
            raise_log(ValueError('The model has been trained with static covariates. Some matching static covariates must be embedded in the target `series` passed to `predict()`.'))

    def _predict_wrapper(self, n: int, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]], future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]], num_samples: int, verbose: bool=False, predict_likelihood_parameters: bool=False) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if False:
            return 10
        kwargs = dict()
        if self.supports_likelihood_parameter_prediction:
            kwargs['predict_likelihood_parameters'] = predict_likelihood_parameters
        return self.predict(n, series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=num_samples, verbose=verbose, **kwargs)

    def _fit_wrapper(self, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]], future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]):
        if False:
            i = 10
            return i + 15
        self.fit(series=series, past_covariates=past_covariates if self.supports_past_covariates else None, future_covariates=future_covariates if self.supports_future_covariates else None)

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        if False:
            print('Hello World!')
        'GlobalForecastingModel supports historical forecasts without retraining the model'
        return True

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        if False:
            return 10
        '\n        Whether the model supports optimized historical forecasts\n        '
        return True

    def _sanity_check_predict_likelihood_parameters(self, n: int, output_chunk_length: Union[int, None], num_samples: int):
        if False:
            for i in range(10):
                print('nop')
        'Verify that the assumptions for likelihood parameters prediction are verified:\n        - Probabilistic models fitted with a likelihood\n        - `num_samples=1`\n        - `n <= output_chunk_length`\n        '
        if not self.supports_likelihood_parameter_prediction:
            raise_log(ValueError('`predict_likelihood_parameters=True` is only supported for probabilistic models fitted with a likelihood.'), logger)
        if num_samples != 1:
            raise_log(ValueError(f'`predict_likelihood_parameters=True` is only supported for `num_samples=1`, received {num_samples}.'), logger)
        if output_chunk_length is not None and n > output_chunk_length:
            raise_log(ValueError('`predict_likelihood_parameters=True` is only supported for `n` smaller than or equal to `output_chunk_length`.'), logger)

class FutureCovariatesLocalForecastingModel(LocalForecastingModel, ABC):
    """The base class for future covariates "local" forecasting models, handling single uni- or multivariate target
    and optional future covariates time series.

    Future Covariates Local Forecasting Models (FC-LFM) are models that can be trained on a single uni- or multivariate
    target and optional future covariates series. In Darts, most models in this category tend to be simpler statistical
    models (such as ARIMA). FC-LFMs usually train on the entire target and future covariates series supplied when
    calling :func:`fit()` at once. They can also predict in one go with :func:`predict()` for any number of predictions
    `n` after the end of the training series. When using future covariates, the values for the future `n` prediction
    steps must be given in the covariate series.

    All implementations must implement the :func:`_fit()` and :func:`_predict()` methods.
    """

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            print('Hello World!')
        'Fit/train the model on the (single) provided series.\n\n        Optionally, a future covariates series can be provided as well.\n\n        Parameters\n        ----------\n        series\n            The model will be trained to forecast this time series. Can be multivariate if the model supports it.\n        future_covariates\n            A time series of future-known covariates. This time series will not be forecasted, but can be used by\n            some models as an input. It must contain at least the same time steps/indices as the target `series`.\n            If it is longer than necessary, it will be automatically trimmed.\n\n        Returns\n        -------\n        self\n            Fitted model.\n        '
        if future_covariates is not None:
            if not series.has_same_time_as(future_covariates):
                future_covariates = future_covariates.slice_intersect(series)
            raise_if_not(series.has_same_time_as(future_covariates), 'The provided `future_covariates` series must contain at least the same time steps/indices as the target `series`.', logger=logger)
            self._expect_future_covariates = True
        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            (_, future_covariates) = self.generate_fit_encodings(series=series, past_covariates=None, future_covariates=future_covariates)
        super().fit(series)
        return self._fit(series, future_covariates=future_covariates)

    @abstractmethod
    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            print('Hello World!')
        'Fits/trains the model on the provided series.\n        DualCovariatesModels must implement the fit logic in this method.\n        '
        pass

    def predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, **kwargs) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        'Forecasts values for `n` time steps after the end of the training series.\n\n        If some future covariates were specified during the training, they must also be specified here.\n\n        Parameters\n        ----------\n        n\n            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.\n        future_covariates\n            The time series of future-known covariates which can be fed as input to the model. It must correspond to\n            the covariate time series that has been used with the :func:`fit()` method for training, and it must\n            contain at least the next `n` time steps/indices after the end of the training target series.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n\n        Returns\n        -------\n        TimeSeries, a single time series containing the `n` next points after then end of the training series.\n        '
        super().predict(n, num_samples)
        if not self._supress_generate_predict_encoding:
            self._verify_passed_predict_covariates(future_covariates)
            if self.encoders is not None and self.encoders.encoding_available:
                (_, future_covariates) = self.generate_predict_encodings(n=n, series=self.training_series, past_covariates=None, future_covariates=future_covariates)
        if future_covariates is not None:
            start = self.training_series.end_time() + self.training_series.freq
            invalid_time_span_error = f'For the given forecasting horizon `n={n}`, the provided `future_covariates` series must contain at least the next `n={n}` time steps/indices after the end of the target `series` that was used to train the model.'
            raise_if_not(future_covariates.end_time() >= start, invalid_time_span_error, logger)
            offset = n - 1 if isinstance(future_covariates.time_index, pd.DatetimeIndex) else n
            future_covariates = future_covariates.slice(start, start + offset * self.training_series.freq)
            raise_if_not(len(future_covariates) == n, invalid_time_span_error, logger)
        return self._predict(n, future_covariates=future_covariates, num_samples=num_samples, **kwargs)

    @abstractmethod
    def _predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False, **kwargs) -> TimeSeries:
        if False:
            while True:
                i = 10
        'Forecasts values for a certain number of time steps after the end of the series.\n        DualCovariatesModels must implement the predict logic in this method.\n        '
        pass

    def _fit_wrapper(self, series: TimeSeries, past_covariates: Optional[TimeSeries], future_covariates: Optional[TimeSeries]):
        if False:
            print('Hello World!')
        self.fit(series, future_covariates=future_covariates)

    def _predict_wrapper(self, n: int, series: TimeSeries, past_covariates: Optional[TimeSeries], future_covariates: Optional[TimeSeries], num_samples: int, verbose: bool=False, predict_likelihood_parameters: bool=False) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        kwargs = dict()
        if self.supports_likelihood_parameter_prediction:
            kwargs['predict_likelihood_parameters'] = predict_likelihood_parameters
        return self.predict(n, future_covariates=future_covariates, num_samples=num_samples, verbose=verbose, **kwargs)

    @property
    def _model_encoder_settings(self) -> Tuple[Optional[int], Optional[int], bool, bool, Optional[List[int]], Optional[List[int]]]:
        if False:
            while True:
                i = 10
        return (None, None, False, True, None, None)

    def _verify_passed_predict_covariates(self, future_covariates):
        if False:
            return 10
        'Simple check if user supplied/did not supply covariates as done at fitting time.'
        if self._expect_future_covariates and future_covariates is None:
            raise_log(ValueError('The model has been trained with `future_covariates` variable. Some matching `future_covariates` variables have to be provided to `predict()`.'))
        if not self._expect_future_covariates and future_covariates is not None:
            raise_log(ValueError('The model has been trained without `future_covariates` variable, but the `future_covariates` parameter provided to `predict()` is not None.'))

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        if False:
            print('Hello World!')
        'Controls wether encodings should be generated in :func:`FutureCovariatesLocalForecastingModel.predict()``'
        return False

    @property
    def extreme_lags(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if False:
            while True:
                i = 10
        return (-self.min_train_series_length, -1, None, None, 0, 0)

class TransferableFutureCovariatesLocalForecastingModel(FutureCovariatesLocalForecastingModel, ABC):
    """The base class for transferable future covariates "local" forecasting models, handling single uni- or
    multivariate target and optional future covariates time series. Additionally, at prediction time, it can be
    applied to new data unrelated to the original series used for fitting the model.

    Transferable Future Covariates Local Forecasting Models (TFC-LFM) are models that can be trained on a single uni-
    or multivariate target and optional future covariates series. Additionally, at prediction time, it can be applied
    to new data unrelated to the original series used for fitting the model. Currently in Darts, all models in this
    category wrap to statsmodel models such as VARIMA. TFC-LFMs usually train on the entire target and future covariates
    series supplied when calling :func:`fit()` at once. They can also predict in one go with :func:`predict()`
    for any number of predictions `n` after the end of the training series. When using future covariates, the values
    for the future `n` prediction steps must be given in the covariate series.

    All implementations must implement the :func:`_fit()` and :func:`_predict()` methods.
    """

    def predict(self, n: int, series: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, **kwargs) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        'If the `series` parameter is not set, forecasts values for `n` time steps after the end of the training\n        series. If some future covariates were specified during the training, they must also be specified here.\n\n        If the `series` parameter is set, forecasts values for `n` time steps after the end of the new target\n        series. If some future covariates were specified during the training, they must also be specified here.\n\n        Parameters\n        ----------\n        n\n            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.\n        series\n            Optionally, a new target series whose future values will be predicted. Defaults to `None`, meaning that the\n            model will forecast the future value of the training series.\n        future_covariates\n            The time series of future-known covariates which can be fed as input to the model. It must correspond to\n            the covariate time series that has been used with the :func:`fit()` method for training.\n\n            If `series` is not set, it must contain at least the next `n` time steps/indices after the end of the\n            training target series. If `series` is set, it must contain at least the time steps/indices corresponding\n            to the new target series (historic future covariates), plus the next `n` time steps/indices after the end.\n        num_samples\n            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1\n            for deterministic models.\n\n        Returns\n        -------\n        TimeSeries, a single time series containing the `n` next points after then end of the training series.\n        '
        self._verify_passed_predict_covariates(future_covariates)
        if self.encoders is not None and self.encoders.encoding_available:
            (_, future_covariates) = self.generate_predict_encodings(n=n, series=series if series is not None else self.training_series, past_covariates=None, future_covariates=future_covariates)
        historic_future_covariates = None
        if series is not None and future_covariates:
            raise_if_not(future_covariates.start_time() <= series.start_time() and future_covariates.end_time() >= series.end_time() + n * series.freq, 'The provided `future_covariates` related to the new target series must contain at least the same timesteps/indices as the target `series` + `n`.', logger)
            (historic_future_covariates, future_covariates) = future_covariates.split_after(series.end_time())
            if not series.has_same_time_as(historic_future_covariates):
                historic_future_covariates = historic_future_covariates.slice_intersect(series)
        if series is not None:
            self._orig_training_series = self.training_series
            self.training_series = series
        result = super().predict(n=n, series=series, historic_future_covariates=historic_future_covariates, future_covariates=future_covariates, num_samples=num_samples, **kwargs)
        if series is not None:
            self.training_series = self._orig_training_series
        return result

    def generate_predict_encodings(self, n: int, series: Union[TimeSeries, Sequence[TimeSeries]], past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]:
        if False:
            print('Hello World!')
        raise_if(self.encoders is None or not self.encoders.encoding_available, 'Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the model with `model.fit()` before.', logger=logger)
        return self.generate_fit_predict_encodings(n=n, series=series, past_covariates=past_covariates, future_covariates=future_covariates)

    @abstractmethod
    def _predict(self, n: int, series: Optional[TimeSeries]=None, historic_future_covariates: Optional[TimeSeries]=None, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False) -> TimeSeries:
        if False:
            print('Hello World!')
        'Forecasts values for a certain number of time steps after the end of the series.\n        TransferableFutureCovariatesLocalForecastingModel must implement the predict logic in this method.\n        '
        pass

    def _predict_wrapper(self, n: int, series: TimeSeries, past_covariates: Optional[TimeSeries], future_covariates: Optional[TimeSeries], num_samples: int, verbose: bool=False, predict_likelihood_parameters: bool=False) -> TimeSeries:
        if False:
            return 10
        kwargs = dict()
        if self.supports_likelihood_parameter_prediction:
            kwargs['predict_likelihood_parameters'] = predict_likelihood_parameters
        return self.predict(n=n, series=series, future_covariates=future_covariates, num_samples=num_samples, verbose=verbose, **kwargs)

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        if False:
            return 10
        return True

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True