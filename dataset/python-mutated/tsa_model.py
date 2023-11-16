from __future__ import annotations
from statsmodels.compat.pandas import is_float_index, is_int_index, is_numeric_dtype
import numbers
import warnings
import numpy as np
from pandas import DatetimeIndex, Index, Period, PeriodIndex, RangeIndex, Series, Timestamp, date_range, period_range, to_datetime
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
_tsa_doc = "\n    %(model)s\n\n    Parameters\n    ----------\n    %(params)s\n    dates : array_like, optional\n        An array-like object of datetime objects. If a pandas object is given\n        for endog or exog, it is assumed to have a DateIndex.\n    freq : str, optional\n        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',\n        'M', 'A', or 'Q'. This is optional if dates are given.\n    %(extra_params)s\n    %(extra_sections)s"
_model_doc = 'Timeseries model base class'
_generic_params = base._model_params_doc
_missing_param_doc = base._missing_param_doc

def get_index_loc(key, index):
    if False:
        while True:
            i = 10
    '\n    Get the location of a specific key in an index\n\n    Parameters\n    ----------\n    key : label\n        The key for which to find the location if the underlying index is\n        a DateIndex or a location if the underlying index is a RangeIndex\n        or an Index with an integer dtype.\n    index : pd.Index\n        The index to search.\n\n    Returns\n    -------\n    loc : int\n        The location of the key\n    index : pd.Index\n        The index including the key; this is a copy of the original index\n        unless the index had to be expanded to accommodate `key`.\n    index_was_expanded : bool\n        Whether or not the index was expanded to accommodate `key`.\n\n    Notes\n    -----\n    If `key` is past the end of of the given index, and the index is either\n    an Index with an integral dtype or a date index, this function extends\n    the index up to and including key, and then returns the location in the\n    new index.\n    '
    base_index = index
    index = base_index
    date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
    int_index = is_int_index(base_index)
    range_index = isinstance(base_index, RangeIndex)
    index_class = type(base_index)
    nobs = len(index)
    if range_index and isinstance(key, (int, np.integer)):
        if key < 0 and -key <= nobs:
            key = nobs + key
        elif key > nobs - 1:
            try:
                base_index_start = base_index.start
                base_index_step = base_index.step
            except AttributeError:
                base_index_start = base_index._start
                base_index_step = base_index._step
            stop = base_index_start + (key + 1) * base_index_step
            index = RangeIndex(start=base_index_start, stop=stop, step=base_index_step)
    if not range_index and int_index and (not date_index) and isinstance(key, (int, np.integer)):
        if key < 0 and -key <= nobs:
            key = nobs + key
        elif key > base_index[-1]:
            index = Index(np.arange(base_index[0], int(key + 1)))
    if date_index:
        if index_class is DatetimeIndex:
            index_fn = date_range
        else:
            index_fn = period_range
        if isinstance(key, (int, np.integer)):
            if key < 0 and -key < nobs:
                key = index[nobs + key]
            elif key > len(base_index) - 1:
                index = index_fn(start=base_index[0], periods=int(key + 1), freq=base_index.freq)
                key = index[-1]
            else:
                key = index[key]
        else:
            if index_class is PeriodIndex:
                date_key = Period(key, freq=base_index.freq)
            else:
                date_key = Timestamp(key)
            if date_key > base_index[-1]:
                index = index_fn(start=base_index[0], end=date_key, freq=base_index.freq)
                if not index[-1] == date_key:
                    index = index_fn(start=base_index[0], periods=len(index) + 1, freq=base_index.freq)
                key = index[-1]
    if date_index:
        loc = index.get_loc(key)
    elif int_index or range_index:
        try:
            index[key]
        except (IndexError, ValueError) as e:
            raise KeyError(str(e))
        loc = key
    else:
        loc = index.get_loc(key)
    index_was_expanded = index is not base_index
    if isinstance(loc, slice):
        end = loc.stop - 1
    else:
        end = loc
    return (loc, index[:end + 1], index_was_expanded)

def get_index_label_loc(key, index, row_labels):
    if False:
        print('Hello World!')
    "\n    Get the location of a specific key in an index or model row labels\n\n    Parameters\n    ----------\n    key : label\n        The key for which to find the location if the underlying index is\n        a DateIndex or is only being used as row labels, or a location if\n        the underlying index is a RangeIndex or a NumericIndex.\n    index : pd.Index\n        The index to search.\n    row_labels : pd.Index\n        Row labels to search if key not found in index\n\n    Returns\n    -------\n    loc : int\n        The location of the key\n    index : pd.Index\n        The index including the key; this is a copy of the original index\n        unless the index had to be expanded to accommodate `key`.\n    index_was_expanded : bool\n        Whether or not the index was expanded to accommodate `key`.\n\n    Notes\n    -----\n    This function expands on `get_index_loc` by first trying the given\n    base index (or the model's index if the base index was not given) and\n    then falling back to try again with the model row labels as the base\n    index.\n    "
    try:
        (loc, index, index_was_expanded) = get_index_loc(key, index)
    except KeyError as e:
        try:
            if not isinstance(key, (int, np.integer)):
                loc = row_labels.get_loc(key)
            else:
                raise
            if isinstance(loc, slice):
                loc = loc.start
            if isinstance(loc, np.ndarray):
                if loc.dtype == bool:
                    loc = np.argmax(loc)
                else:
                    loc = loc[0]
            if not isinstance(loc, numbers.Integral):
                raise
            index = row_labels[:loc + 1]
            index_was_expanded = False
        except:
            raise e
    return (loc, index, index_was_expanded)

def get_prediction_index(start, end, nobs, base_index, index=None, silent=False, index_none=False, index_generated=None, data=None) -> tuple[int, int, int, Index | None]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the location of a specific key in an index or model row labels\n\n    Parameters\n    ----------\n    start : label\n        The key at which to start prediction. Depending on the underlying\n        model's index, may be an integer, a date (string, datetime object,\n        pd.Timestamp, or pd.Period object), or some other object in the\n        model's row labels.\n    end : label\n        The key at which to end prediction (note that this key will be\n        *included* in prediction). Depending on the underlying\n        model's index, may be an integer, a date (string, datetime object,\n        pd.Timestamp, or pd.Period object), or some other object in the\n        model's row labels.\n    nobs : int\n    base_index : pd.Index\n\n    index : pd.Index, optional\n        Optionally an index to associate the predicted results to. If None,\n        an attempt is made to create an index for the predicted results\n        from the model's index or model's row labels.\n    silent : bool, optional\n        Argument to silence warnings.\n\n    Returns\n    -------\n    start : int\n        The index / observation location at which to begin prediction.\n    end : int\n        The index / observation location at which to end in-sample\n        prediction. The maximum value for this is nobs-1.\n    out_of_sample : int\n        The number of observations to forecast after the end of the sample.\n    prediction_index : pd.Index or None\n        The index associated with the prediction results. This index covers\n        the range [start, end + out_of_sample]. If the model has no given\n        index and no given row labels (i.e. endog/exog is not Pandas), then\n        this will be None.\n\n    Notes\n    -----\n    The arguments `start` and `end` behave differently, depending on if\n    they are integer or not. If either is an integer, then it is assumed\n    to refer to a *location* in the index, not to an index value. On the\n    other hand, if it is a date string or some other type of object, then\n    it is assumed to refer to an index *value*. In all cases, the returned\n    `start` and `end` values refer to index *locations* (so in the former\n    case, the given location is validated and returned whereas in the\n    latter case a location is found that corresponds to the given index\n    value).\n\n    This difference in behavior is necessary to support `RangeIndex`. This\n    is because integers for a RangeIndex could refer either to index values\n    or to index locations in an ambiguous way (while for `NumericIndex`,\n    since we have required them to be full indexes, there is no ambiguity).\n    "
    try:
        (start, _, start_oos) = get_index_label_loc(start, base_index, data.row_labels)
    except KeyError:
        raise KeyError('The `start` argument could not be matched to a location related to the index of the data.')
    if end is None:
        end = max(start, len(base_index) - 1)
    try:
        (end, end_index, end_oos) = get_index_label_loc(end, base_index, data.row_labels)
    except KeyError:
        raise KeyError('The `end` argument could not be matched to a location related to the index of the data.')
    if isinstance(start, slice):
        start = start.start
    if isinstance(end, slice):
        end = end.stop - 1
    prediction_index = end_index[start:]
    if end < start:
        raise ValueError('Prediction must have `end` after `start`.')
    if index is not None:
        if not len(prediction_index) == len(index):
            raise ValueError('Invalid `index` provided in prediction. Must have length consistent with `start` and `end` arguments.')
        if not isinstance(data, PandasData) and (not silent):
            warnings.warn('Because the model data (`endog`, `exog`) were not given as Pandas objects, the prediction output will be Numpy arrays, and the given `index` argument will only be used internally.', ValueWarning, stacklevel=2)
        prediction_index = Index(index)
    elif index_generated and (not index_none):
        if data.row_labels is not None and (not (start_oos or end_oos)):
            prediction_index = data.row_labels[start:end + 1]
        else:
            if not silent:
                warnings.warn('No supported index is available. Prediction results will be given with an integer index beginning at `start`.', ValueWarning, stacklevel=2)
            warnings.warn('No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.', FutureWarning, stacklevel=2)
    elif index_none:
        prediction_index = None
    if prediction_index is not None:
        data.predict_start = prediction_index[0]
        data.predict_end = prediction_index[-1]
        data.predict_dates = prediction_index
    else:
        data.predict_start = None
        data.predict_end = None
        data.predict_dates = None
    out_of_sample = max(end - (nobs - 1), 0)
    end -= out_of_sample
    return (start, end, out_of_sample, prediction_index)

class TimeSeriesModel(base.LikelihoodModel):
    __doc__ = _tsa_doc % {'model': _model_doc, 'params': _generic_params, 'extra_params': _missing_param_doc, 'extra_sections': ''}

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(endog, exog, missing=missing, **kwargs)
        self._init_dates(dates, freq)

    def _init_dates(self, dates=None, freq=None):
        if False:
            i = 10
            return i + 15
        '\n        Initialize dates\n\n        Parameters\n        ----------\n        dates : array_like, optional\n            An array like object containing dates.\n        freq : str, tuple, datetime.timedelta, DateOffset or None, optional\n            A frequency specification for either `dates` or the row labels from\n            the endog / exog data.\n\n        Notes\n        -----\n        Creates `self._index` and related attributes. `self._index` is always\n        a Pandas index, and it is always NumericIndex, DatetimeIndex, or\n        PeriodIndex.\n\n        If Pandas objects, endog / exog may have any type of index. If it is\n        an NumericIndex with values 0, 1, ..., nobs-1 or if it is (coerceable to)\n        a DatetimeIndex or PeriodIndex *with an associated frequency*, then it\n        is called a "supported" index. Otherwise it is called an "unsupported"\n        index.\n\n        Supported indexes are standardized (i.e. a list of date strings is\n        converted to a DatetimeIndex) and the result is put in `self._index`.\n\n        Unsupported indexes are ignored, and a supported NumericIndex is\n        generated and put in `self._index`. Warnings are issued in this case\n        to alert the user if the returned index from some operation (e.g.\n        forecasting) is different from the original data\'s index. However,\n        whenever possible (e.g. purely in-sample prediction), the original\n        index is returned.\n\n        The benefit of supported indexes is that they allow *forecasting*, i.e.\n        it is possible to extend them in a reasonable way. Thus every model\n        must have an underlying supported index, even if it is just a generated\n        NumericIndex.\n        '
        if dates is not None:
            index = dates
        else:
            index = self.data.row_labels
        if index is None and freq is not None:
            raise ValueError('Frequency provided without associated index.')
        inferred_freq = False
        if index is not None:
            if not isinstance(index, (DatetimeIndex, PeriodIndex)):
                try:
                    _index = np.asarray(index)
                    if is_numeric_dtype(_index) or is_float_index(index) or isinstance(_index[0], float):
                        raise ValueError('Numeric index given')
                    if isinstance(index, Series):
                        index = index.values
                    _index = to_datetime(index)
                    if not isinstance(_index, Index):
                        raise ValueError('Could not coerce to date index')
                    index = _index
                except:
                    if dates is not None:
                        raise ValueError('Non-date index index provided to `dates` argument.')
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                if freq is None and index.freq is None:
                    freq = index.inferred_freq
                    if freq is not None:
                        inferred_freq = True
                        if freq is not None:
                            warnings.warn('No frequency information was provided, so inferred frequency %s will be used.' % freq, ValueWarning, stacklevel=2)
                if freq is not None:
                    freq = to_offset(freq)
                if freq is None and index.freq is None:
                    if dates is not None:
                        raise ValueError('No frequency information was provided with date index and no frequency could be inferred.')
                elif freq is not None and index.freq is None:
                    resampled_index = date_range(start=index[0], end=index[-1], freq=freq)
                    if not inferred_freq and (not resampled_index.equals(index)):
                        raise ValueError('The given frequency argument could not be matched to the given index.')
                    index = resampled_index
                elif freq is not None and (not inferred_freq) and (not index.freq == freq):
                    raise ValueError('The given frequency argument is incompatible with the given index.')
            elif freq is not None:
                raise ValueError('Given index could not be coerced to dates but `freq` argument was provided.')
        has_index = index is not None
        date_index = isinstance(index, (DatetimeIndex, PeriodIndex))
        period_index = isinstance(index, PeriodIndex)
        int_index = is_int_index(index)
        range_index = isinstance(index, RangeIndex)
        has_freq = index.freq is not None if date_index else None
        increment = Index(range(self.endog.shape[0]))
        is_increment = index.equals(increment) if int_index else None
        if date_index:
            try:
                is_monotonic = index.is_monotonic_increasing
            except AttributeError:
                is_monotonic = index.is_monotonic
        else:
            is_monotonic = None
        if has_index and (not (date_index or range_index or is_increment)):
            warnings.warn('An unsupported index was provided and will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
        if date_index and (not has_freq):
            warnings.warn('A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
        if date_index and (not is_monotonic):
            warnings.warn('A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.', ValueWarning, stacklevel=2)
        index_generated = False
        valid_index = date_index and has_freq and is_monotonic or (int_index and is_increment) or range_index
        if valid_index:
            _index = index
        else:
            _index = increment
            index_generated = True
        self._index = _index
        self._index_generated = index_generated
        self._index_none = index is None
        self._index_int64 = int_index and (not range_index) and (not date_index)
        self._index_dates = date_index and (not index_generated)
        self._index_freq = self._index.freq if self._index_dates else None
        self._index_inferred_freq = inferred_freq
        self.data.dates = self._index if self._index_dates else None
        self.data.freq = self._index.freqstr if self._index_dates else None

    def _get_index_loc(self, key, base_index=None):
        if False:
            while True:
                i = 10
        "\n        Get the location of a specific key in an index\n\n        Parameters\n        ----------\n        key : label\n            The key for which to find the location if the underlying index is\n            a DateIndex or a location if the underlying index is a RangeIndex\n            or an NumericIndex.\n        base_index : pd.Index, optional\n            Optionally the base index to search. If None, the model's index is\n            searched.\n\n        Returns\n        -------\n        loc : int\n            The location of the key\n        index : pd.Index\n            The index including the key; this is a copy of the original index\n            unless the index had to be expanded to accommodate `key`.\n        index_was_expanded : bool\n            Whether or not the index was expanded to accommodate `key`.\n\n        Notes\n        -----\n        If `key` is past the end of of the given index, and the index is either\n        an NumericIndex or a date index, this function extends the index up to\n        and including key, and then returns the location in the new index.\n        "
        if base_index is None:
            base_index = self._index
        return get_index_loc(key, base_index)

    def _get_index_label_loc(self, key, base_index=None):
        if False:
            return 10
        "\n        Get the location of a specific key in an index or model row labels\n\n        Parameters\n        ----------\n        key : label\n            The key for which to find the location if the underlying index is\n            a DateIndex or is only being used as row labels, or a location if\n            the underlying index is a RangeIndex or an NumericIndex.\n        base_index : pd.Index, optional\n            Optionally the base index to search. If None, the model's index is\n            searched.\n\n        Returns\n        -------\n        loc : int\n            The location of the key\n        index : pd.Index\n            The index including the key; this is a copy of the original index\n            unless the index had to be expanded to accommodate `key`.\n        index_was_expanded : bool\n            Whether or not the index was expanded to accommodate `key`.\n\n        Notes\n        -----\n        This method expands on `_get_index_loc` by first trying the given\n        base index (or the model's index if the base index was not given) and\n        then falling back to try again with the model row labels as the base\n        index.\n        "
        if base_index is None:
            base_index = self._index
        return get_index_label_loc(key, base_index, self.data.row_labels)

    def _get_prediction_index(self, start, end, index=None, silent=False) -> tuple[int, int, int, Index | None]:
        if False:
            i = 10
            return i + 15
        "\n        Get the location of a specific key in an index or model row labels\n\n        Parameters\n        ----------\n        start : label\n            The key at which to start prediction. Depending on the underlying\n            model's index, may be an integer, a date (string, datetime object,\n            pd.Timestamp, or pd.Period object), or some other object in the\n            model's row labels.\n        end : label\n            The key at which to end prediction (note that this key will be\n            *included* in prediction). Depending on the underlying\n            model's index, may be an integer, a date (string, datetime object,\n            pd.Timestamp, or pd.Period object), or some other object in the\n            model's row labels.\n        index : pd.Index, optional\n            Optionally an index to associate the predicted results to. If None,\n            an attempt is made to create an index for the predicted results\n            from the model's index or model's row labels.\n        silent : bool, optional\n            Argument to silence warnings.\n\n        Returns\n        -------\n        start : int\n            The index / observation location at which to begin prediction.\n        end : int\n            The index / observation location at which to end in-sample\n            prediction. The maximum value for this is nobs-1.\n        out_of_sample : int\n            The number of observations to forecast after the end of the sample.\n        prediction_index : pd.Index or None\n            The index associated with the prediction results. This index covers\n            the range [start, end + out_of_sample]. If the model has no given\n            index and no given row labels (i.e. endog/exog is not Pandas), then\n            this will be None.\n\n        Notes\n        -----\n        The arguments `start` and `end` behave differently, depending on if\n        they are integer or not. If either is an integer, then it is assumed\n        to refer to a *location* in the index, not to an index value. On the\n        other hand, if it is a date string or some other type of object, then\n        it is assumed to refer to an index *value*. In all cases, the returned\n        `start` and `end` values refer to index *locations* (so in the former\n        case, the given location is validated and returned whereas in the\n        latter case a location is found that corresponds to the given index\n        value).\n\n        This difference in behavior is necessary to support `RangeIndex`. This\n        is because integers for a RangeIndex could refer either to index values\n        or to index locations in an ambiguous way (while for `NumericIndex`,\n        since we have required them to be full indexes, there is no ambiguity).\n        "
        nobs = len(self.endog)
        return get_prediction_index(start, end, nobs, base_index=self._index, index=index, silent=silent, index_none=self._index_none, index_generated=self._index_generated, data=self.data)

    def _get_exog_names(self):
        if False:
            return 10
        return self.data.xnames

    def _set_exog_names(self, vals):
        if False:
            while True:
                i = 10
        if not isinstance(vals, list):
            vals = [vals]
        self.data.xnames = vals
    exog_names = property(_get_exog_names, _set_exog_names, None, 'The names of the exogenous variables.')

class TimeSeriesModelResults(base.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params, scale=1.0):
        if False:
            print('Hello World!')
        self.data = model.data
        super().__init__(model, params, normalized_cov_params, scale)

class TimeSeriesResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates'}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(TimeSeriesResultsWrapper, TimeSeriesModelResults)