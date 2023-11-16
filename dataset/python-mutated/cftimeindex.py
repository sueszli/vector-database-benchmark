"""DatetimeIndex analog for cftime.datetime objects"""
from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import _STANDARD_CALENDARS, cftime_to_nptime, infer_calendar_name
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
try:
    import cftime
except ImportError:
    cftime = None
CFTIME_REPR_LENGTH = 19
ITEMS_IN_REPR_MAX_ELSE_ELLIPSIS = 100
REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END = 10
OUT_OF_BOUNDS_TIMEDELTA_ERRORS: tuple[type[Exception], ...]
try:
    OUT_OF_BOUNDS_TIMEDELTA_ERRORS = (pd.errors.OutOfBoundsTimedelta, OverflowError)
except AttributeError:
    OUT_OF_BOUNDS_TIMEDELTA_ERRORS = (OverflowError,)

def named(name, pattern):
    if False:
        i = 10
        return i + 15
    return '(?P<' + name + '>' + pattern + ')'

def optional(x):
    if False:
        for i in range(10):
            print('nop')
    return '(?:' + x + ')?'

def trailing_optional(xs):
    if False:
        while True:
            i = 10
    if not xs:
        return ''
    return xs[0] + optional(trailing_optional(xs[1:]))

def build_pattern(date_sep='\\-', datetime_sep='T', time_sep='\\:'):
    if False:
        return 10
    pieces = [(None, 'year', '\\d{4}'), (date_sep, 'month', '\\d{2}'), (date_sep, 'day', '\\d{2}'), (datetime_sep, 'hour', '\\d{2}'), (time_sep, 'minute', '\\d{2}'), (time_sep, 'second', '\\d{2}')]
    pattern_list = []
    for (sep, name, sub_pattern) in pieces:
        pattern_list.append((sep if sep else '') + named(name, sub_pattern))
    return '^' + trailing_optional(pattern_list) + '$'
_BASIC_PATTERN = build_pattern(date_sep='', time_sep='')
_EXTENDED_PATTERN = build_pattern()
_CFTIME_PATTERN = build_pattern(datetime_sep=' ')
_PATTERNS = [_BASIC_PATTERN, _EXTENDED_PATTERN, _CFTIME_PATTERN]

def parse_iso8601_like(datetime_string):
    if False:
        while True:
            i = 10
    for pattern in _PATTERNS:
        match = re.match(pattern, datetime_string)
        if match:
            return match.groupdict()
    raise ValueError(f'no ISO-8601 or cftime-string-like match for string: {datetime_string}')

def _parse_iso8601_with_reso(date_type, timestr):
    if False:
        while True:
            i = 10
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    default = date_type(1, 1, 1)
    result = parse_iso8601_like(timestr)
    replace = {}
    for attr in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        value = result.get(attr, None)
        if value is not None:
            replace[attr] = int(value)
            resolution = attr
    return (default.replace(**replace), resolution)

def _parsed_string_to_bounds(date_type, resolution, parsed):
    if False:
        print('Hello World!')
    'Generalization of\n    pandas.tseries.index.DatetimeIndex._parsed_string_to_bounds\n    for use with non-standard calendars and cftime.datetime\n    objects.\n    '
    if resolution == 'year':
        return (date_type(parsed.year, 1, 1), date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1))
    elif resolution == 'month':
        if parsed.month == 12:
            end = date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = date_type(parsed.year, parsed.month + 1, 1) - timedelta(microseconds=1)
        return (date_type(parsed.year, parsed.month, 1), end)
    elif resolution == 'day':
        start = date_type(parsed.year, parsed.month, parsed.day)
        return (start, start + timedelta(days=1, microseconds=-1))
    elif resolution == 'hour':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour)
        return (start, start + timedelta(hours=1, microseconds=-1))
    elif resolution == 'minute':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute)
        return (start, start + timedelta(minutes=1, microseconds=-1))
    elif resolution == 'second':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
        return (start, start + timedelta(seconds=1, microseconds=-1))
    else:
        raise KeyError

def get_date_field(datetimes, field):
    if False:
        for i in range(10):
            print('nop')
    'Adapted from pandas.tslib.get_date_field'
    return np.array([getattr(date, field) for date in datetimes])

def _field_accessor(name, docstring=None, min_cftime_version='0.0'):
    if False:
        print('Hello World!')
    'Adapted from pandas.tseries.index._field_accessor'

    def f(self, min_cftime_version=min_cftime_version):
        if False:
            while True:
                i = 10
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if Version(cftime.__version__) >= Version(min_cftime_version):
            return get_date_field(self._data, name)
        else:
            raise ImportError(f'The {name:!r} accessor requires a minimum version of cftime of {min_cftime_version}. Found an installed version of {cftime.__version__}.')
    f.__name__ = name
    f.__doc__ = docstring
    return property(f)

def get_date_type(self):
    if False:
        for i in range(10):
            print('nop')
    if self._data.size:
        return type(self._data[0])
    else:
        return None

def assert_all_valid_date_type(data):
    if False:
        i = 10
        return i + 15
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if len(data) > 0:
        sample = data[0]
        date_type = type(sample)
        if not isinstance(sample, cftime.datetime):
            raise TypeError(f'CFTimeIndex requires cftime.datetime objects. Got object of {date_type}.')
        if not all((isinstance(value, date_type) for value in data)):
            raise TypeError(f'CFTimeIndex requires using datetime objects of all the same type.  Got\n{data}.')

def format_row(times, indent=0, separator=', ', row_end=',\n'):
    if False:
        print('Hello World!')
    'Format a single row from format_times.'
    return indent * ' ' + separator.join(map(str, times)) + row_end

def format_times(index, max_width, offset, separator=', ', first_row_offset=0, intermediate_row_end=',\n', last_row_end=''):
    if False:
        while True:
            i = 10
    'Format values of cftimeindex as pd.Index.'
    n_per_row = max(max_width // (CFTIME_REPR_LENGTH + len(separator)), 1)
    n_rows = math.ceil(len(index) / n_per_row)
    representation = ''
    for row in range(n_rows):
        indent = first_row_offset if row == 0 else offset
        row_end = last_row_end if row == n_rows - 1 else intermediate_row_end
        times_for_row = index[row * n_per_row:(row + 1) * n_per_row]
        representation += format_row(times_for_row, indent=indent, separator=separator, row_end=row_end)
    return representation

def format_attrs(index, separator=', '):
    if False:
        return 10
    'Format attributes of CFTimeIndex for __repr__.'
    attrs = {'dtype': f"'{index.dtype}'", 'length': f'{len(index)}', 'calendar': f"'{index.calendar}'", 'freq': f"'{index.freq}'" if len(index) >= 3 else None}
    attrs_str = [f'{k}={v}' for (k, v) in attrs.items()]
    attrs_str = f'{separator}'.join(attrs_str)
    return attrs_str

class CFTimeIndex(pd.Index):
    """Custom Index for working with CF calendars and dates

    All elements of a CFTimeIndex must be cftime.datetime objects.

    Parameters
    ----------
    data : array or CFTimeIndex
        Sequence of cftime.datetime objects to use in index
    name : str, default: None
        Name of the resulting index

    See Also
    --------
    cftime_range
    """
    year = _field_accessor('year', 'The year of the datetime')
    month = _field_accessor('month', 'The month of the datetime')
    day = _field_accessor('day', 'The days of the datetime')
    hour = _field_accessor('hour', 'The hours of the datetime')
    minute = _field_accessor('minute', 'The minutes of the datetime')
    second = _field_accessor('second', 'The seconds of the datetime')
    microsecond = _field_accessor('microsecond', 'The microseconds of the datetime')
    dayofyear = _field_accessor('dayofyr', 'The ordinal day of year of the datetime', '1.0.2.1')
    dayofweek = _field_accessor('dayofwk', 'The day of week of the datetime', '1.0.2.1')
    days_in_month = _field_accessor('daysinmonth', 'The number of days in the month of the datetime', '1.1.0.0')
    date_type = property(get_date_type)

    def __new__(cls, data, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert_all_valid_date_type(data)
        if name is None and hasattr(data, 'name'):
            name = data.name
        result = object.__new__(cls)
        result._data = np.array(data, dtype='O')
        result.name = name
        result._cache = {}
        return result

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Return a string representation for this object.\n        '
        klass_name = type(self).__name__
        display_width = OPTIONS['display_width']
        offset = len(klass_name) + 2
        if len(self) <= ITEMS_IN_REPR_MAX_ELSE_ELLIPSIS:
            datastr = format_times(self.values, display_width, offset=offset, first_row_offset=0)
        else:
            front_str = format_times(self.values[:REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END], display_width, offset=offset, first_row_offset=0, last_row_end=',')
            end_str = format_times(self.values[-REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END:], display_width, offset=offset, first_row_offset=offset)
            datastr = '\n'.join([front_str, f"{' ' * offset}...", end_str])
        attrs_str = format_attrs(self)
        full_repr_str = f'{klass_name}([{datastr}], {attrs_str})'
        if len(full_repr_str) > display_width:
            if len(attrs_str) >= display_width - offset:
                attrs_str = attrs_str.replace(',', f",\n{' ' * (offset - 2)}")
            full_repr_str = f"{klass_name}([{datastr}],\n{' ' * (offset - 1)}{attrs_str})"
        return full_repr_str

    def _partial_date_slice(self, resolution, parsed):
        if False:
            for i in range(10):
                print('nop')
        'Adapted from\n        pandas.tseries.index.DatetimeIndex._partial_date_slice\n\n        Note that when using a CFTimeIndex, if a partial-date selection\n        returns a single element, it will never be converted to a scalar\n        coordinate; this is in slight contrast to the behavior when using\n        a DatetimeIndex, which sometimes will return a DataArray with a scalar\n        coordinate depending on the resolution of the datetimes used in\n        defining the index.  For example:\n\n        >>> from cftime import DatetimeNoLeap\n        >>> da = xr.DataArray(\n        ...     [1, 2],\n        ...     coords=[[DatetimeNoLeap(2001, 1, 1), DatetimeNoLeap(2001, 2, 1)]],\n        ...     dims=["time"],\n        ... )\n        >>> da.sel(time="2001-01-01")\n        <xarray.DataArray (time: 1)>\n        array([1])\n        Coordinates:\n          * time     (time) object 2001-01-01 00:00:00\n        >>> da = xr.DataArray(\n        ...     [1, 2],\n        ...     coords=[[pd.Timestamp(2001, 1, 1), pd.Timestamp(2001, 2, 1)]],\n        ...     dims=["time"],\n        ... )\n        >>> da.sel(time="2001-01-01")\n        <xarray.DataArray ()>\n        array(1)\n        Coordinates:\n            time     datetime64[ns] 2001-01-01\n        >>> da = xr.DataArray(\n        ...     [1, 2],\n        ...     coords=[[pd.Timestamp(2001, 1, 1, 1), pd.Timestamp(2001, 2, 1)]],\n        ...     dims=["time"],\n        ... )\n        >>> da.sel(time="2001-01-01")\n        <xarray.DataArray (time: 1)>\n        array([1])\n        Coordinates:\n          * time     (time) datetime64[ns] 2001-01-01T01:00:00\n        '
        (start, end) = _parsed_string_to_bounds(self.date_type, resolution, parsed)
        times = self._data
        if self.is_monotonic_increasing:
            if len(times) and (start < times[0] and end < times[0] or (start > times[-1] and end > times[-1])):
                raise KeyError
            left = times.searchsorted(start, side='left')
            right = times.searchsorted(end, side='right')
            return slice(left, right)
        lhs_mask = times >= start
        rhs_mask = times <= end
        return np.flatnonzero(lhs_mask & rhs_mask)

    def _get_string_slice(self, key):
        if False:
            print('Hello World!')
        'Adapted from pandas.tseries.index.DatetimeIndex._get_string_slice'
        (parsed, resolution) = _parse_iso8601_with_reso(self.date_type, key)
        try:
            loc = self._partial_date_slice(resolution, parsed)
        except KeyError:
            raise KeyError(key)
        return loc

    def _get_nearest_indexer(self, target, limit, tolerance):
        if False:
            while True:
                i = 10
        'Adapted from pandas.Index._get_nearest_indexer'
        left_indexer = self.get_indexer(target, 'pad', limit=limit)
        right_indexer = self.get_indexer(target, 'backfill', limit=limit)
        left_distances = abs(self.values[left_indexer] - target.values)
        right_distances = abs(self.values[right_indexer] - target.values)
        if self.is_monotonic_increasing:
            condition = (left_distances < right_distances) | (right_indexer == -1)
        else:
            condition = (left_distances <= right_distances) | (right_indexer == -1)
        indexer = np.where(condition, left_indexer, right_indexer)
        if tolerance is not None:
            indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
        return indexer

    def _filter_indexer_tolerance(self, target, indexer, tolerance):
        if False:
            return 10
        'Adapted from pandas.Index._filter_indexer_tolerance'
        if isinstance(target, pd.Index):
            distance = abs(self.values[indexer] - target.values)
        else:
            distance = abs(self.values[indexer] - target)
        indexer = np.where(distance <= tolerance, indexer, -1)
        return indexer

    def get_loc(self, key):
        if False:
            while True:
                i = 10
        'Adapted from pandas.tseries.index.DatetimeIndex.get_loc'
        if isinstance(key, str):
            return self._get_string_slice(key)
        else:
            return super().get_loc(key)

    def _maybe_cast_slice_bound(self, label, side):
        if False:
            i = 10
            return i + 15
        'Adapted from\n        pandas.tseries.index.DatetimeIndex._maybe_cast_slice_bound\n        '
        if not isinstance(label, str):
            return label
        (parsed, resolution) = _parse_iso8601_with_reso(self.date_type, label)
        (start, end) = _parsed_string_to_bounds(self.date_type, resolution, parsed)
        if self.is_monotonic_decreasing and len(self) > 1:
            return end if side == 'left' else start
        return start if side == 'left' else end

    def get_value(self, series, key):
        if False:
            for i in range(10):
                print('nop')
        'Adapted from pandas.tseries.index.DatetimeIndex.get_value'
        if np.asarray(key).dtype == np.dtype(bool):
            return series.iloc[key]
        elif isinstance(key, slice):
            return series.iloc[self.slice_indexer(key.start, key.stop, key.step)]
        else:
            return series.iloc[self.get_loc(key)]

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Adapted from\n        pandas.tseries.base.DatetimeIndexOpsMixin.__contains__'
        try:
            result = self.get_loc(key)
            return is_scalar(result) or type(result) == slice or (isinstance(result, np.ndarray) and result.size)
        except (KeyError, TypeError, ValueError):
            return False

    def contains(self, key):
        if False:
            print('Hello World!')
        'Needed for .loc based partial-string indexing'
        return self.__contains__(key)

    def shift(self, n: int | float, freq: str | timedelta):
        if False:
            print('Hello World!')
        'Shift the CFTimeIndex a multiple of the given frequency.\n\n        See the documentation for :py:func:`~xarray.cftime_range` for a\n        complete listing of valid frequency strings.\n\n        Parameters\n        ----------\n        n : int, float if freq of days or below\n            Periods to shift by\n        freq : str or datetime.timedelta\n            A frequency string or datetime.timedelta object to shift by\n\n        Returns\n        -------\n        CFTimeIndex\n\n        See Also\n        --------\n        pandas.DatetimeIndex.shift\n\n        Examples\n        --------\n        >>> index = xr.cftime_range("2000", periods=1, freq="M")\n        >>> index\n        CFTimeIndex([2000-01-31 00:00:00],\n                    dtype=\'object\', length=1, calendar=\'standard\', freq=None)\n        >>> index.shift(1, "M")\n        CFTimeIndex([2000-02-29 00:00:00],\n                    dtype=\'object\', length=1, calendar=\'standard\', freq=None)\n        >>> index.shift(1.5, "D")\n        CFTimeIndex([2000-02-01 12:00:00],\n                    dtype=\'object\', length=1, calendar=\'standard\', freq=None)\n        '
        if isinstance(freq, timedelta):
            return self + n * freq
        elif isinstance(freq, str):
            from xarray.coding.cftime_offsets import to_offset
            return self + n * to_offset(freq)
        else:
            raise TypeError(f"'freq' must be of type str or datetime.timedelta, got {freq}.")

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, pd.TimedeltaIndex):
            other = other.to_pytimedelta()
        return CFTimeIndex(np.array(self) + other)

    def __radd__(self, other):
        if False:
            return 10
        if isinstance(other, pd.TimedeltaIndex):
            other = other.to_pytimedelta()
        return CFTimeIndex(other + np.array(self))

    def __sub__(self, other):
        if False:
            return 10
        if _contains_datetime_timedeltas(other):
            return CFTimeIndex(np.array(self) - other)
        elif isinstance(other, pd.TimedeltaIndex):
            return CFTimeIndex(np.array(self) - other.to_pytimedelta())
        elif _contains_cftime_datetimes(np.array(other)):
            try:
                return pd.TimedeltaIndex(np.array(self) - np.array(other))
            except OUT_OF_BOUNDS_TIMEDELTA_ERRORS:
                raise ValueError('The time difference exceeds the range of values that can be expressed at the nanosecond resolution.')
        else:
            return NotImplemented

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            return pd.TimedeltaIndex(other - np.array(self))
        except OUT_OF_BOUNDS_TIMEDELTA_ERRORS:
            raise ValueError('The time difference exceeds the range of values that can be expressed at the nanosecond resolution.')

    def to_datetimeindex(self, unsafe=False):
        if False:
            i = 10
            return i + 15
        'If possible, convert this index to a pandas.DatetimeIndex.\n\n        Parameters\n        ----------\n        unsafe : bool\n            Flag to turn off warning when converting from a CFTimeIndex with\n            a non-standard calendar to a DatetimeIndex (default ``False``).\n\n        Returns\n        -------\n        pandas.DatetimeIndex\n\n        Raises\n        ------\n        ValueError\n            If the CFTimeIndex contains dates that are not possible in the\n            standard calendar or outside the nanosecond-precision range.\n\n        Warns\n        -----\n        RuntimeWarning\n            If converting from a non-standard calendar to a DatetimeIndex.\n\n        Warnings\n        --------\n        Note that for non-standard calendars, this will change the calendar\n        type of the index.  In that case the result of this method should be\n        used with caution.\n\n        Examples\n        --------\n        >>> times = xr.cftime_range("2000", periods=2, calendar="gregorian")\n        >>> times\n        CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00],\n                    dtype=\'object\', length=2, calendar=\'standard\', freq=None)\n        >>> times.to_datetimeindex()\n        DatetimeIndex([\'2000-01-01\', \'2000-01-02\'], dtype=\'datetime64[ns]\', freq=None)\n        '
        nptimes = cftime_to_nptime(self)
        calendar = infer_calendar_name(self)
        if calendar not in _STANDARD_CALENDARS and (not unsafe):
            warnings.warn(f'Converting a CFTimeIndex with dates from a non-standard calendar, {calendar!r}, to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.', RuntimeWarning, stacklevel=2)
        return pd.DatetimeIndex(nptimes)

    def strftime(self, date_format):
        if False:
            while True:
                i = 10
        '\n        Return an Index of formatted strings specified by date_format, which\n        supports the same string format as the python standard library. Details\n        of the string format can be found in `python string format doc\n        <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__\n\n        Parameters\n        ----------\n        date_format : str\n            Date format string (e.g. "%Y-%m-%d")\n\n        Returns\n        -------\n        pandas.Index\n            Index of formatted strings\n\n        Examples\n        --------\n        >>> rng = xr.cftime_range(\n        ...     start="2000", periods=5, freq="2MS", calendar="noleap"\n        ... )\n        >>> rng.strftime("%B %d, %Y, %r")\n        Index([\'January 01, 2000, 12:00:00 AM\', \'March 01, 2000, 12:00:00 AM\',\n               \'May 01, 2000, 12:00:00 AM\', \'July 01, 2000, 12:00:00 AM\',\n               \'September 01, 2000, 12:00:00 AM\'],\n              dtype=\'object\')\n        '
        return pd.Index([date.strftime(date_format) for date in self._data])

    @property
    def asi8(self):
        if False:
            return 10
        'Convert to integers with units of microseconds since 1970-01-01.'
        from xarray.core.resample_cftime import exact_cftime_datetime_difference
        epoch = self.date_type(1970, 1, 1)
        return np.array([_total_microseconds(exact_cftime_datetime_difference(epoch, date)) for date in self.values], dtype=np.int64)

    @property
    def calendar(self):
        if False:
            return 10
        'The calendar used by the datetimes in the index.'
        from xarray.coding.times import infer_calendar_name
        return infer_calendar_name(self)

    @property
    def freq(self):
        if False:
            print('Hello World!')
        'The frequency used by the dates in the index.'
        from xarray.coding.frequencies import infer_freq
        return infer_freq(self)

    def _round_via_method(self, freq, method):
        if False:
            print('Hello World!')
        'Round dates using a specified method.'
        from xarray.coding.cftime_offsets import CFTIME_TICKS, to_offset
        offset = to_offset(freq)
        if not isinstance(offset, CFTIME_TICKS):
            raise ValueError(f'{offset} is a non-fixed frequency')
        unit = _total_microseconds(offset.as_timedelta())
        values = self.asi8
        rounded = method(values, unit)
        return _cftimeindex_from_i8(rounded, self.date_type, self.name)

    def floor(self, freq):
        if False:
            i = 10
            return i + 15
        "Round dates down to fixed frequency.\n\n        Parameters\n        ----------\n        freq : str\n            The frequency level to round the index to.  Must be a fixed\n            frequency like 'S' (second) not 'ME' (month end).  See `frequency\n            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_\n            for a list of possible values.\n\n        Returns\n        -------\n        CFTimeIndex\n        "
        return self._round_via_method(freq, _floor_int)

    def ceil(self, freq):
        if False:
            while True:
                i = 10
        "Round dates up to fixed frequency.\n\n        Parameters\n        ----------\n        freq : str\n            The frequency level to round the index to.  Must be a fixed\n            frequency like 'S' (second) not 'ME' (month end).  See `frequency\n            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_\n            for a list of possible values.\n\n        Returns\n        -------\n        CFTimeIndex\n        "
        return self._round_via_method(freq, _ceil_int)

    def round(self, freq):
        if False:
            return 10
        "Round dates to a fixed frequency.\n\n        Parameters\n        ----------\n        freq : str\n            The frequency level to round the index to.  Must be a fixed\n            frequency like 'S' (second) not 'ME' (month end).  See `frequency\n            aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_\n            for a list of possible values.\n\n        Returns\n        -------\n        CFTimeIndex\n        "
        return self._round_via_method(freq, _round_to_nearest_half_even)

def _parse_iso8601_without_reso(date_type, datetime_str):
    if False:
        return 10
    (date, _) = _parse_iso8601_with_reso(date_type, datetime_str)
    return date

def _parse_array_of_cftime_strings(strings, date_type):
    if False:
        i = 10
        return i + 15
    'Create a numpy array from an array of strings.\n\n    For use in generating dates from strings for use with interp.  Assumes the\n    array is either 0-dimensional or 1-dimensional.\n\n    Parameters\n    ----------\n    strings : array of strings\n        Strings to convert to dates\n    date_type : cftime.datetime type\n        Calendar type to use for dates\n\n    Returns\n    -------\n    np.array\n    '
    return np.array([_parse_iso8601_without_reso(date_type, s) for s in strings.ravel()]).reshape(strings.shape)

def _contains_datetime_timedeltas(array):
    if False:
        for i in range(10):
            print('nop')
    'Check if an input array contains datetime.timedelta objects.'
    array = np.atleast_1d(array)
    return isinstance(array[0], timedelta)

def _cftimeindex_from_i8(values, date_type, name):
    if False:
        i = 10
        return i + 15
    'Construct a CFTimeIndex from an array of integers.\n\n    Parameters\n    ----------\n    values : np.array\n        Integers representing microseconds since 1970-01-01.\n    date_type : cftime.datetime\n        Type of date for the index.\n    name : str\n        Name of the index.\n\n    Returns\n    -------\n    CFTimeIndex\n    '
    epoch = date_type(1970, 1, 1)
    dates = np.array([epoch + timedelta(microseconds=int(value)) for value in values])
    return CFTimeIndex(dates, name=name)

def _total_microseconds(delta):
    if False:
        while True:
            i = 10
    'Compute the total number of microseconds of a datetime.timedelta.\n\n    Parameters\n    ----------\n    delta : datetime.timedelta\n        Input timedelta.\n\n    Returns\n    -------\n    int\n    '
    return delta / timedelta(microseconds=1)

def _floor_int(values, unit):
    if False:
        for i in range(10):
            print('nop')
    'Copied from pandas.'
    return values - np.remainder(values, unit)

def _ceil_int(values, unit):
    if False:
        return 10
    'Copied from pandas.'
    return values + np.remainder(-values, unit)

def _round_to_nearest_half_even(values, unit):
    if False:
        for i in range(10):
            print('nop')
    'Copied from pandas.'
    if unit % 2:
        return _ceil_int(values - unit // 2, unit)
    (quotient, remainder) = np.divmod(values, unit)
    mask = np.logical_or(remainder > unit // 2, np.logical_and(remainder == unit // 2, quotient % 2))
    quotient[mask] += 1
    return quotient * unit