"""
Facebook Prophet
----------------
"""
import logging
import re
from typing import Callable, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import prophet
from darts.logging import execute_and_suppress_output, get_logger, raise_if, raise_log
from darts.models.forecasting.forecasting_model import FutureCovariatesLocalForecastingModel
from darts.timeseries import TimeSeries
logger = get_logger(__name__)
logger.level = logging.WARNING

class Prophet(FutureCovariatesLocalForecastingModel):

    def __init__(self, add_seasonalities: Optional[Union[dict, List[dict]]]=None, country_holidays: Optional[str]=None, suppress_stdout_stderror: bool=True, add_encoders: Optional[dict]=None, cap: Optional[Union[float, Callable[[Union[pd.DatetimeIndex, pd.RangeIndex]], Sequence[float]]]]=None, floor: Optional[Union[float, Callable[[Union[pd.DatetimeIndex, pd.RangeIndex]], Sequence[float]]]]=None, **prophet_kwargs):
        if False:
            while True:
                i = 10
        'Facebook Prophet\n\n        This class provides a basic wrapper around `Facebook Prophet <https://github.com/facebook/prophet>`_.\n        It supports adding country holidays as well as custom seasonalities and adds support for stochastic\n        forecasting and future covariates.\n\n        Parameters\n        ----------\n        add_seasonalities\n            Optionally, a dict or list of dicts with custom seasonality/ies to add to the model.\n            Each dict takes the following mandatory and optional data:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                dict({\n                \'name\': str  # (name of the seasonality component),\n                \'seasonal_periods\': Union[int, float]  # (nr of steps composing a season),\n                \'fourier_order\': int  # (number of Fourier components to use),\n                \'prior_scale\': Optional[float]  # (a prior scale for this component),\n                \'mode\': Optional[str]  # (\'additive\' or \'multiplicative\')\n                })\n            ..\n\n            An example for `seasonal_periods`: If you have hourly data (frequency=\'H\') and your seasonal cycle repeats\n            after 48 hours then set `seasonal_periods=48`. Notice that this value will be multiplied by the inferred\n            number of days for the TimeSeries frequency (1 / 24 in this example) to be consistent with the\n            `add_seasonality()` method of Facebook Prophet, where the `period` parameter is specified in days.\n\n            Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet\'s\n            `add_seasonality()` method.\n            Alternatively, you can add seasonalities after model creation and before fitting with\n            :meth:`add_seasonality() <Prophet.add_seasonality()>`.\n        country_holidays\n            An optional country code, for which holidays can be taken into account by Prophet.\n\n            See: https://github.com/dr-prodigy/python-holidays\n\n            In addition to those countries, Prophet includes holidays for these\n            countries: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN),\n            Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD),\n            Egypt (EG), China (CN), and Russia (RU).\n        suppress_stdout_stderror\n            Optionally suppress the log output produced by Prophet during training.\n        add_encoders\n            A large number of future covariates can be automatically generated with `add_encoders`.\n            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that\n            will be used as index encoders. Additionally, a transformer such as Darts\' :class:`Scaler` can be added to\n            transform the generated covariates. This happens all under one hood and only needs to be specified at\n            model creation.\n            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about\n            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:\n\n            .. highlight:: python\n            .. code-block:: python\n\n                def encode_year(idx):\n                    return (idx.year - 1950) / 50\n\n                add_encoders={\n                    \'cyclic\': {\'future\': [\'month\']},\n                    \'datetime_attribute\': {\'future\': [\'hour\', \'dayofweek\']},\n                    \'position\': {\'future\': [\'relative\']},\n                    \'custom\': {\'future\': [encode_year]},\n                    \'transformer\': Scaler(),\n                    \'tz\': \'CET\'\n                }\n            ..\n        cap\n            Parameter specifiying the maximum carrying capacity when predicting with logistic growth.\n            Mandatory when `growth = \'logistic\'`, otherwise ignored.\n            See <https://facebook.github.io/prophet/docs/saturating_forecasts.html> for more information\n            on logistic forecasts.\n            Can be either\n\n            - a number, for constant carrying capacities\n            - a function taking a DatetimeIndex or RangeIndex and returning a corresponding a Sequence of numbers,\n            where each number indicates the carrying capacity at this index.\n        floor\n            Parameter specifiying the minimum carrying capacity when predicting logistic growth.\n            Optional when `growth = \'logistic\'` (defaults to 0), otherwise ignored.\n            See <https://facebook.github.io/prophet/docs/saturating_forecasts.html> for more information\n            on logistic forecasts.\n            Can be either\n\n            - a number, for constant carrying capacities\n            - a function taking a DatetimeIndex or RangeIndex and returning a corresponding a Sequence of numbers,\n            where each number indicates the carrying capacity at this index.\n        prophet_kwargs\n            Some optional keyword arguments for Prophet.\n            For information about the parameters see:\n            `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.models import Prophet\n        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries\n        >>> series = AirPassengersDataset().load()\n        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series\n        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)\n        >>> # adding a seasonality (daily, weekly and yearly are included by default) and holidays\n        >>> model = Prophet(\n        >>>     add_seasonalities={\n        >>>         \'name\':"quarterly_seasonality",\n        >>>         \'seasonal_periods\':4,\n        >>>         \'fourier_order\':5\n        >>>         },\n        >>> )\n        >>> model.fit(series, future_covariates=future_cov)\n        >>> pred = model.predict(6)\n        >>> pred.values()\n        array([[472.26891239],\n               [467.56955721],\n               [494.47230467],\n               [493.10568429],\n               [497.54686113],\n               [539.11716811]])\n        '
        super().__init__(add_encoders=add_encoders)
        self._auto_seasonalities = self._extract_auto_seasonality(prophet_kwargs)
        self._add_seasonalities = dict()
        add_seasonality_calls = add_seasonalities if isinstance(add_seasonalities, list) else [add_seasonalities]
        for call in add_seasonality_calls:
            self._store_add_seasonality_call(seasonality_call=call)
        self.country_holidays = country_holidays
        self.prophet_kwargs = prophet_kwargs
        self.model = None
        self.suppress_stdout_stderr = suppress_stdout_stderror
        self._execute_and_suppress_output = execute_and_suppress_output
        self._model_builder = prophet.Prophet
        self._cap = cap
        self._floor = floor
        self.is_logistic = 'growth' in prophet_kwargs and prophet_kwargs['growth'] == 'logistic'
        if not self.is_logistic and (cap is not None or floor is not None):
            logger.warning('Parameters `cap` and/or `floor` were set although `growth` is not logistic. The set capacities will be ignored.')
        if self.is_logistic:
            raise_if(cap is None, 'Parameter `cap` has to be set when `growth` is logistic', logger)
            if floor is None:
                self._floor = 0

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries]=None):
        if False:
            return 10
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series
        fit_df = pd.DataFrame(data={'ds': series.time_index, 'y': series.univariate_values()})
        if self.is_logistic:
            fit_df = self._add_capacities_to_df(fit_df)
        self.model = self._model_builder(**self.prophet_kwargs)
        interval_length = self._freq_to_days(series.freq_str)
        conditional_seasonality_covariates = self._check_seasonality_conditions(future_covariates=future_covariates)
        for (seasonality_name, attributes) in self._add_seasonalities.items():
            self.model.add_seasonality(name=seasonality_name, period=attributes['seasonal_periods'] * interval_length, fourier_order=attributes['fourier_order'], prior_scale=attributes['prior_scale'], mode=attributes['mode'], condition_name=attributes['condition_name'])
        if future_covariates is not None:
            fit_df = fit_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
            for covariate in future_covariates.columns:
                if covariate not in conditional_seasonality_covariates:
                    self.model.add_regressor(covariate)
        if self.country_holidays is not None:
            self.model.add_country_holidays(self.country_holidays)
        if self.suppress_stdout_stderr:
            self._execute_and_suppress_output(self.model.fit, logger, logging.WARNING, fit_df)
        else:
            self.model.fit(fit_df)
        return self

    def _predict(self, n: int, future_covariates: Optional[TimeSeries]=None, num_samples: int=1, verbose: bool=False) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        _ = self._check_seasonality_conditions(future_covariates=future_covariates)
        super()._predict(n, future_covariates, num_samples)
        predict_df = self._generate_predict_df(n=n, future_covariates=future_covariates)
        if num_samples == 1:
            forecast = self.model.predict(predict_df, vectorized=True)['yhat'].values
        else:
            forecast = np.expand_dims(self._stochastic_samples(predict_df, n_samples=num_samples), axis=1)
        return self._build_forecast_series(forecast)

    def _add_capacities_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        dates = df['ds']
        try:
            df['cap'] = self._cap(dates) if callable(self._cap) else self._cap
            df['floor'] = self._floor(dates) if callable(self._floor) else self._floor
        except ValueError as e:
            raise_if('does not match length of index' in str(e), 'Callables supplied to `Prophet.set_capacity` as `cap` or `floor` arguments have to return Sequences of identical length as their  input argument Sequence!', logger)
            raise
        return df

    def _generate_predict_df(self, n: int, future_covariates: Optional[TimeSeries]=None) -> pd.DataFrame:
        if False:
            return 10
        'Returns a pandas DataFrame in the format required for Prophet.predict() with `n` dates after the end of\n        the fitted TimeSeries'
        predict_df = pd.DataFrame(data={'ds': self._generate_new_dates(n)})
        if self.is_logistic:
            predict_df = self._add_capacities_to_df(predict_df)
        if future_covariates is not None:
            predict_df = predict_df.merge(future_covariates.pd_dataframe(), left_on='ds', right_index=True, how='left')
        return predict_df

    def _check_seasonality_conditions(self, future_covariates: Optional[TimeSeries]=None) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the conditions for custom conditional seasonalities are met. Each custom seasonality that has a\n        `condition_name` other than None is checked. If the `condition_name` is not a column in the `future_covariates`\n        or if the values in the column are not all True or False, an error is raised.\n        Returns a list of the `condition_name`s of the conditional seasonalities that have been checked.\n\n        Parameters\n        ----------\n        future_covariates\n            optionally, a TimeSeries containing the future covariates and including the columns that are used as\n            conditions for the conditional seasonalities when necessary\n\n        Raises\n        ------\n        ValueError\n            if a seasonality has a `condition_name` and a column named `condition_name` is missing in\n            the `future_covariates`\n\n            if a seasonality has a `condition_name` and the values in the corresponding column in `future_covariates`\n            are not binary values (True or False, 1 or 0)\n        '
        conditional_seasonality_covariates = []
        invalid_conditional_seasonalities = []
        if future_covariates is not None:
            future_covariates_columns = future_covariates.columns
        else:
            future_covariates_columns = []
        for (seasonality_name, attributes) in self._add_seasonalities.items():
            condition_name = attributes['condition_name']
            if condition_name is not None:
                if condition_name not in future_covariates_columns:
                    invalid_conditional_seasonalities.append((seasonality_name, condition_name, 'column missing'))
                    continue
                if not future_covariates[condition_name].pd_series().isin([True, False]).all():
                    invalid_conditional_seasonalities.append((seasonality_name, condition_name, 'invalid values'))
                    continue
                conditional_seasonality_covariates.append(condition_name)
        if len(invalid_conditional_seasonalities) > 0:
            formatted_issues_str = ', '.join((f"'{name}' (condition_name: '{cond}'; issue: {reason})" for (name, cond, reason) in invalid_conditional_seasonalities))
            raise_log(ValueError(f'The following seasonalities have invalid conditions: {formatted_issues_str}. Each conditional seasonality must be accompanied by a binary component/column in the `future_covariates` with the same name as the `condition_name`'), logger)
        return conditional_seasonality_covariates

    @property
    def supports_multivariate(self) -> bool:
        if False:
            return 10
        return False

    @property
    def _is_probabilistic(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def _stochastic_samples(self, predict_df, n_samples) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Returns stochastic forecast of `n_samples` samples.\n        This method is a replicate of Prophet.predict() which suspends simplification of stochastic samples to\n        deterministic target values.'
        n_samples_default = self.model.uncertainty_samples
        self.model.uncertainty_samples = n_samples
        if self.model.history is None:
            raise ValueError('Model has not been fit.')
        if predict_df is None:
            predict_df = self.model.history.copy()
        else:
            if predict_df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            predict_df = self.model.setup_dataframe(predict_df.copy())
        predict_df['trend'] = self.model.predict_trend(predict_df)
        forecast = self.model.sample_posterior_predictive(predict_df, vectorized=True)
        self.model.uncertainty_samples = n_samples_default
        return forecast['yhat']

    def predict_raw(self, n: int, future_covariates: Optional[TimeSeries]=None) -> pd.DataFrame:
        if False:
            return 10
        'Returns the output of the base Facebook Prophet model in form of a pandas DataFrame. Note however,\n        that the output of this method is not supported for further processing with the Darts API.\n\n        Methods of the base Prophet model can be accessed with self.model.method() (i.e. self.model.plot_components())\n        '
        super().predict(n, future_covariates, num_samples=1)
        predict_df = self._generate_predict_df(n=n, future_covariates=future_covariates)
        return self.model.predict(predict_df, vectorized=True)

    def add_seasonality(self, name: str, seasonal_periods: Union[int, float], fourier_order: int, prior_scale: Optional[float]=None, mode: Optional[str]=None, condition_name: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        "Adds a custom seasonality to the model that repeats after every n `seasonal_periods` timesteps.\n        An example for `seasonal_periods`: If you have hourly data (frequency='H') and your seasonal cycle repeats\n        after 48 hours -> `seasonal_periods=48`.\n\n        Apart from `seasonal_periods`, this is very similar to how you would call Facebook Prophet's\n        `add_seasonality()` method.\n\n        To add conditional seasonalities, provide `condition_name` here, and add a boolean (binary) component/column\n        named `condition_name` to the `future_covariates` series passed to `fit()` and `predict()`.\n\n        For information about the parameters see:\n        `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`.\n        For more details on conditional seasonalities see:\n        https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#seasonalities-that-depend-on-other-factors\n\n        Parameters\n        ----------\n        name\n            name of the seasonality component\n        seasonal_periods\n            number of timesteps after which the seasonal cycle repeats. This value will be multiplied by the inferred\n            number of days for the TimeSeries frequency (e.g. 365.25 for a yearly frequency) to be consistent with the\n            `add_seasonality()` method of Facebook Prophet. The inferred number of days can be obtained with\n            `model._freq_to_days(series.freq)`, where `model` is the `Prophet` model and `series` is the target series.\n        fourier_order\n            number of Fourier components to use\n        prior_scale\n            optionally, a prior scale for this component\n        mode\n            optionally, 'additive' or 'multiplicative'\n        condition_name\n            optionally, the name of the condition on which the seasonality depends. If not `None`, expects a\n            `future_covariates` time series with a component/column named `condition_name` to be passed to `fit()`\n            and `predict()`.\n        "
        function_call = {'name': name, 'seasonal_periods': seasonal_periods, 'fourier_order': fourier_order, 'prior_scale': prior_scale, 'mode': mode, 'condition_name': condition_name}
        self._store_add_seasonality_call(seasonality_call=function_call)

    def _store_add_seasonality_call(self, seasonality_call: Optional[dict]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Checks the validity of an add_seasonality() call and stores valid calls.\n        As the actual model is only created at fitting time, and seasonalities are added pre-fit,\n        the add_seasonality calls must be stored and checked on Darts' side.\n\n        Raises\n        ----------\n        ValueError\n            if `seasonality_call` has missing or empty mandatory keys/arguments\n\n            if `seasonality_call` with `name` already exists.\n\n            if `seasonality_call` has invalid keys/arguments\n\n            if `seasonality_call` has invalid dtypes\n        "
        if seasonality_call is None:
            return
        seasonality_properties = {'name': {'default': None, 'dtype': str}, 'seasonal_periods': {'default': None, 'dtype': (int, float)}, 'fourier_order': {'default': None, 'dtype': int}, 'prior_scale': {'default': None, 'dtype': float}, 'mode': {'default': None, 'dtype': str}, 'condition_name': {'default': None, 'dtype': str}}
        seasonality_default = {kw: seasonality_properties[kw]['default'] for kw in seasonality_properties}
        mandatory_keywords = ['name', 'seasonal_periods', 'fourier_order']
        add_seasonality_call = dict(seasonality_default, **seasonality_call)
        missing_kws = [kw for kw in mandatory_keywords if add_seasonality_call[kw] is None]
        raise_if(len(missing_kws) > 0, f"Seasonality `{add_seasonality_call['name']}` has missing mandatory keywords or empty arguments: {missing_kws}.", logger)
        seasonality_name = add_seasonality_call['name']
        raise_if(seasonality_name in self._auto_seasonalities or seasonality_name in self._add_seasonalities, f'Adding seasonality with `name={seasonality_name}` failed. A seasonality with this name already exists.')
        invalid_kws = [kw for kw in add_seasonality_call if kw not in seasonality_default]
        raise_if(len(invalid_kws) > 0, f"Seasonality `{add_seasonality_call['name']}` has invalid keywords: {invalid_kws}. Only the following arguments are supported: {list(seasonality_default)}", logger)
        invalid_types = [kw for (kw, value) in add_seasonality_call.items() if not isinstance(value, seasonality_properties[kw]['dtype']) and value is not None]
        raise_if(len(invalid_types) > 0, f"Seasonality `{add_seasonality_call['name']}` has invalid value dtypes: {invalid_types} must be of type {[seasonality_properties[kw]['dtype'] for kw in invalid_types]}.", logger)
        self._add_seasonalities[seasonality_name] = add_seasonality_call

    @staticmethod
    def _extract_auto_seasonality(prophet_kwargs: dict) -> list:
        if False:
            print('Hello World!')
        "Returns the automatically added seasonalities by Prophet's base model based on kwargs of model creation"
        auto_seasonalities = []
        for auto_seasonality in ['daily', 'weekly', 'yearly']:
            s_name = auto_seasonality + '_seasonality'
            if not (s_name in prophet_kwargs and (not prophet_kwargs[s_name])):
                auto_seasonalities.append(auto_seasonality)
        return auto_seasonalities

    @staticmethod
    def _freq_to_days(freq: str) -> float:
        if False:
            i = 10
            return i + 15
        "Converts a frequency to number of days required by Facebook Prophet\n\n        Parameters\n        ----------\n        freq\n            frequency string of the underlying TimeSeries's time index (pd.DateTimeIndex.freq_str)\n        "
        freq_times = re.findall('\\d+', freq)
        freq_times = 1 if not freq_times else int(freq_times[0])
        freq = ''.join(re.split('[^a-zA-Z-]*', freq)).split('-')[0]
        seconds_per_day = 86400
        if freq in ['A', 'BA', 'Y', 'BY', 'RE'] or freq.startswith(('A', 'BA', 'Y', 'BY', 'RE')):
            days = 365.25
        elif freq in ['Q', 'BQ', 'REQ'] or freq.startswith(('Q', 'BQ', 'REQ')):
            days = 3 * 30.4375
        elif freq in ['M', 'BM', 'CBM', 'SM'] or freq.startswith(('M', 'BM', 'BS', 'CBM', 'SM')):
            days = 30.4375
        elif freq in ['W']:
            days = 7.0
        elif freq in ['B', 'C']:
            days = 1 * 7 / 5
        elif freq in ['D']:
            days = 1.0
        elif freq in ['H', 'BH', 'CBH']:
            days = 1 / 24
        elif freq in ['T', 'min']:
            days = 1 / (24 * 60)
        elif freq in ['S']:
            days = 1 / seconds_per_day
        elif freq in ['L', 'ms']:
            days = 1 / (seconds_per_day * 10 ** 3)
        elif freq in ['U', 'us']:
            days = 1 / (seconds_per_day * 10 ** 6)
        elif freq in ['N']:
            days = 1 / (seconds_per_day * 10 ** 9)
        else:
            raise ValueError('freq {} not understood. Please report if you think this is in error.'.format(freq))
        return freq_times * days

    @property
    def _supports_range_index(self) -> bool:
        if False:
            return 10
        'Prophet does not support integer range index.'
        raise_if(True, 'Prophet does not support integer range index. The index of the TimeSeries must be of type pandas.DatetimeIndex', logger)
        return False