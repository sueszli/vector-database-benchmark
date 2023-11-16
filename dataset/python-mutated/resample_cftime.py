"""Resampling for CFTimeIndex. Does not support non-integer freq."""
from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import BaseCFTimeOffset, MonthEnd, QuarterEnd, Tick, YearEnd, cftime_range, normalize_date, to_offset
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
if typing.TYPE_CHECKING:
    from xarray.core.types import CFTimeDatetime

class CFTimeGrouper:
    """This is a simple container for the grouping parameters that implements a
    single method, the only one required for resampling in xarray.  It cannot
    be used in a call to groupby like a pandas.Grouper object can."""

    def __init__(self, freq: str | BaseCFTimeOffset, closed: SideOptions | None=None, label: SideOptions | None=None, loffset: str | datetime.timedelta | BaseCFTimeOffset | None=None, origin: str | CFTimeDatetime='start_day', offset: str | datetime.timedelta | None=None):
        if False:
            i = 10
            return i + 15
        self.offset: datetime.timedelta | None
        self.closed: SideOptions
        self.label: SideOptions
        self.freq = to_offset(freq)
        self.loffset = loffset
        self.origin = origin
        if isinstance(self.freq, (MonthEnd, QuarterEnd, YearEnd)):
            if closed is None:
                self.closed = 'right'
            else:
                self.closed = closed
            if label is None:
                self.label = 'right'
            else:
                self.label = label
        elif self.origin in ['end', 'end_day']:
            if closed is None:
                self.closed = 'right'
            else:
                self.closed = closed
            if label is None:
                self.label = 'right'
            else:
                self.label = label
        else:
            if closed is None:
                self.closed = 'left'
            else:
                self.closed = closed
            if label is None:
                self.label = 'left'
            else:
                self.label = label
        if offset is not None:
            try:
                self.offset = _convert_offset_to_timedelta(offset)
            except (ValueError, AttributeError) as error:
                raise ValueError(f'offset must be a datetime.timedelta object or an offset string that can be converted to a timedelta.  Got {offset} instead.') from error
        else:
            self.offset = None

    def first_items(self, index: CFTimeIndex):
        if False:
            for i in range(10):
                print('nop')
        'Meant to reproduce the results of the following\n\n        grouper = pandas.Grouper(...)\n        first_items = pd.Series(np.arange(len(index)),\n                                index).groupby(grouper).first()\n\n        with index being a CFTimeIndex instead of a DatetimeIndex.\n        '
        (datetime_bins, labels) = _get_time_bins(index, self.freq, self.closed, self.label, self.origin, self.offset)
        if self.loffset is not None:
            if not isinstance(self.loffset, (str, datetime.timedelta, BaseCFTimeOffset)):
                raise ValueError(f'`loffset` must be a str or datetime.timedelta object. Got {self.loffset}.')
            if isinstance(self.loffset, datetime.timedelta):
                labels = labels + self.loffset
            else:
                labels = labels + to_offset(self.loffset)
        if index[0] < datetime_bins[0]:
            raise ValueError('Value falls before first bin')
        if index[-1] > datetime_bins[-1]:
            raise ValueError('Value falls after last bin')
        integer_bins = np.searchsorted(index, datetime_bins, side=self.closed)
        counts = np.diff(integer_bins)
        codes = np.repeat(np.arange(len(labels)), counts)
        first_items = pd.Series(integer_bins[:-1], labels, copy=False)
        non_duplicate = ~first_items.duplicated('last')
        return (first_items.where(non_duplicate), codes)

def _get_time_bins(index: CFTimeIndex, freq: BaseCFTimeOffset, closed: SideOptions, label: SideOptions, origin: str | CFTimeDatetime, offset: datetime.timedelta | None):
    if False:
        return 10
    "Obtain the bins and their respective labels for resampling operations.\n\n    Parameters\n    ----------\n    index : CFTimeIndex\n        Index object to be resampled (e.g., CFTimeIndex named 'time').\n    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset\n        The offset object representing target conversion a.k.a. resampling\n        frequency (e.g., 'MS', '2D', 'H', or '3T' with\n        coding.cftime_offsets.to_offset() applied to it).\n    closed : 'left' or 'right'\n        Which side of bin interval is closed.\n        The default is 'left' for all frequency offsets except for 'M' and 'A',\n        which have a default of 'right'.\n    label : 'left' or 'right'\n        Which bin edge label to label bucket with.\n        The default is 'left' for all frequency offsets except for 'M' and 'A',\n        which have a default of 'right'.\n    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'\n        The datetime on which to adjust the grouping. The timezone of origin\n        must match the timezone of the index.\n\n        If a datetime is not used, these values are also supported:\n        - 'epoch': `origin` is 1970-01-01\n        - 'start': `origin` is the first value of the timeseries\n        - 'start_day': `origin` is the first day at midnight of the timeseries\n        - 'end': `origin` is the last value of the timeseries\n        - 'end_day': `origin` is the ceiling midnight of the last day\n    offset : datetime.timedelta, default is None\n        An offset timedelta added to the origin.\n\n    Returns\n    -------\n    datetime_bins : CFTimeIndex\n        Defines the edge of resampling bins by which original index values will\n        be grouped into.\n    labels : CFTimeIndex\n        Define what the user actually sees the bins labeled as.\n    "
    if not isinstance(index, CFTimeIndex):
        raise TypeError(f'index must be a CFTimeIndex, but got an instance of {type(index).__name__!r}')
    if len(index) == 0:
        datetime_bins = labels = CFTimeIndex(data=[], name=index.name)
        return (datetime_bins, labels)
    (first, last) = _get_range_edges(index.min(), index.max(), freq, closed=closed, origin=origin, offset=offset)
    datetime_bins = labels = cftime_range(freq=freq, start=first, end=last, name=index.name)
    (datetime_bins, labels) = _adjust_bin_edges(datetime_bins, freq, closed, index, labels)
    labels = labels[1:] if label == 'right' else labels[:-1]
    return (datetime_bins, labels)

def _adjust_bin_edges(datetime_bins: np.ndarray, freq: BaseCFTimeOffset, closed: SideOptions, index: CFTimeIndex, labels: np.ndarray):
    if False:
        while True:
            i = 10
    "This is required for determining the bin edges resampling with\n    month end, quarter end, and year end frequencies.\n\n    Consider the following example.  Let's say you want to downsample the\n    time series with the following coordinates to month end frequency:\n\n    CFTimeIndex([2000-01-01 12:00:00, 2000-01-31 12:00:00,\n                 2000-02-01 12:00:00], dtype='object')\n\n    Without this adjustment, _get_time_bins with month-end frequency will\n    return the following index for the bin edges (default closed='right' and\n    label='right' in this case):\n\n    CFTimeIndex([1999-12-31 00:00:00, 2000-01-31 00:00:00,\n                 2000-02-29 00:00:00], dtype='object')\n\n    If 2000-01-31 is used as a bound for a bin, the value on\n    2000-01-31T12:00:00 (at noon on January 31st), will not be included in the\n    month of January.  To account for this, pandas adds a day minus one worth\n    of microseconds to the bin edges generated by cftime range, so that we do\n    bin the value at noon on January 31st in the January bin.  This results in\n    an index with bin edges like the following:\n\n    CFTimeIndex([1999-12-31 23:59:59, 2000-01-31 23:59:59,\n                 2000-02-29 23:59:59], dtype='object')\n\n    The labels are still:\n\n    CFTimeIndex([2000-01-31 00:00:00, 2000-02-29 00:00:00], dtype='object')\n    "
    if isinstance(freq, (MonthEnd, QuarterEnd, YearEnd)):
        if closed == 'right':
            datetime_bins = datetime_bins + datetime.timedelta(days=1, microseconds=-1)
        if datetime_bins[-2] > index.max():
            datetime_bins = datetime_bins[:-1]
            labels = labels[:-1]
    return (datetime_bins, labels)

def _get_range_edges(first: CFTimeDatetime, last: CFTimeDatetime, freq: BaseCFTimeOffset, closed: SideOptions='left', origin: str | CFTimeDatetime='start_day', offset: datetime.timedelta | None=None):
    if False:
        for i in range(10):
            print('nop')
    "Get the correct starting and ending datetimes for the resampled\n    CFTimeIndex range.\n\n    Parameters\n    ----------\n    first : cftime.datetime\n        Uncorrected starting datetime object for resampled CFTimeIndex range.\n        Usually the min of the original CFTimeIndex.\n    last : cftime.datetime\n        Uncorrected ending datetime object for resampled CFTimeIndex range.\n        Usually the max of the original CFTimeIndex.\n    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset\n        The offset object representing target conversion a.k.a. resampling\n        frequency. Contains information on offset type (e.g. Day or 'D') and\n        offset magnitude (e.g., n = 3).\n    closed : 'left' or 'right'\n        Which side of bin interval is closed. Defaults to 'left'.\n    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'\n        The datetime on which to adjust the grouping. The timezone of origin\n        must match the timezone of the index.\n\n        If a datetime is not used, these values are also supported:\n        - 'epoch': `origin` is 1970-01-01\n        - 'start': `origin` is the first value of the timeseries\n        - 'start_day': `origin` is the first day at midnight of the timeseries\n        - 'end': `origin` is the last value of the timeseries\n        - 'end_day': `origin` is the ceiling midnight of the last day\n    offset : datetime.timedelta, default is None\n        An offset timedelta added to the origin.\n\n    Returns\n    -------\n    first : cftime.datetime\n        Corrected starting datetime object for resampled CFTimeIndex range.\n    last : cftime.datetime\n        Corrected ending datetime object for resampled CFTimeIndex range.\n    "
    if isinstance(freq, Tick):
        (first, last) = _adjust_dates_anchored(first, last, freq, closed=closed, origin=origin, offset=offset)
        return (first, last)
    else:
        first = normalize_date(first)
        last = normalize_date(last)
    first = freq.rollback(first) if closed == 'left' else first - freq
    last = last + freq
    return (first, last)

def _adjust_dates_anchored(first: CFTimeDatetime, last: CFTimeDatetime, freq: Tick, closed: SideOptions='right', origin: str | CFTimeDatetime='start_day', offset: datetime.timedelta | None=None):
    if False:
        print('Hello World!')
    "First and last offsets should be calculated from the start day to fix\n    an error cause by resampling across multiple days when a one day period is\n    not a multiple of the frequency.\n    See https://github.com/pandas-dev/pandas/issues/8683\n\n    Parameters\n    ----------\n    first : cftime.datetime\n        A datetime object representing the start of a CFTimeIndex range.\n    last : cftime.datetime\n        A datetime object representing the end of a CFTimeIndex range.\n    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset\n        The offset object representing target conversion a.k.a. resampling\n        frequency. Contains information on offset type (e.g. Day or 'D') and\n        offset magnitude (e.g., n = 3).\n    closed : 'left' or 'right'\n        Which side of bin interval is closed. Defaults to 'right'.\n    origin : {'epoch', 'start', 'start_day', 'end', 'end_day'} or cftime.datetime, default 'start_day'\n        The datetime on which to adjust the grouping. The timezone of origin\n        must match the timezone of the index.\n\n        If a datetime is not used, these values are also supported:\n        - 'epoch': `origin` is 1970-01-01\n        - 'start': `origin` is the first value of the timeseries\n        - 'start_day': `origin` is the first day at midnight of the timeseries\n        - 'end': `origin` is the last value of the timeseries\n        - 'end_day': `origin` is the ceiling midnight of the last day\n    offset : datetime.timedelta, default is None\n        An offset timedelta added to the origin.\n\n    Returns\n    -------\n    fresult : cftime.datetime\n        A datetime object representing the start of a date range that has been\n        adjusted to fix resampling errors.\n    lresult : cftime.datetime\n        A datetime object representing the end of a date range that has been\n        adjusted to fix resampling errors.\n    "
    import cftime
    if origin == 'start_day':
        origin_date = normalize_date(first)
    elif origin == 'start':
        origin_date = first
    elif origin == 'epoch':
        origin_date = type(first)(1970, 1, 1)
    elif origin in ['end', 'end_day']:
        origin_last = last if origin == 'end' else _ceil_via_cftimeindex(last, 'D')
        sub_freq_times = (origin_last - first) // freq.as_timedelta()
        if closed == 'left':
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq
        origin_date = first
    elif isinstance(origin, cftime.datetime):
        origin_date = origin
    else:
        raise ValueError(f"origin must be one of {{'epoch', 'start_day', 'start', 'end', 'end_day'}} or a cftime.datetime object.  Got {origin}.")
    if offset is not None:
        origin_date = origin_date + offset
    foffset = (first - origin_date) % freq.as_timedelta()
    loffset = (last - origin_date) % freq.as_timedelta()
    if closed == 'right':
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first - freq.as_timedelta()
        if loffset.total_seconds() > 0:
            lresult = last + (freq.as_timedelta() - loffset)
        else:
            lresult = last
    else:
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first
        if loffset.total_seconds() > 0:
            lresult = last + (freq.as_timedelta() - loffset)
        else:
            lresult = last + freq
    return (fresult, lresult)

def exact_cftime_datetime_difference(a: CFTimeDatetime, b: CFTimeDatetime):
    if False:
        i = 10
        return i + 15
    'Exact computation of b - a\n\n    Assumes:\n\n        a = a_0 + a_m\n        b = b_0 + b_m\n\n    Here a_0, and b_0 represent the input dates rounded\n    down to the nearest second, and a_m, and b_m represent\n    the remaining microseconds associated with date a and\n    date b.\n\n    We can then express the value of b - a as:\n\n        b - a = (b_0 + b_m) - (a_0 + a_m) = b_0 - a_0 + b_m - a_m\n\n    By construction, we know that b_0 - a_0 must be a round number\n    of seconds.  Therefore we can take the result of b_0 - a_0 using\n    ordinary cftime.datetime arithmetic and round to the nearest\n    second.  b_m - a_m is the remainder, in microseconds, and we\n    can simply add this to the rounded timedelta.\n\n    Parameters\n    ----------\n    a : cftime.datetime\n        Input datetime\n    b : cftime.datetime\n        Input datetime\n\n    Returns\n    -------\n    datetime.timedelta\n    '
    seconds = b.replace(microsecond=0) - a.replace(microsecond=0)
    seconds = int(round(seconds.total_seconds()))
    microseconds = b.microsecond - a.microsecond
    return datetime.timedelta(seconds=seconds, microseconds=microseconds)

def _convert_offset_to_timedelta(offset: datetime.timedelta | str | BaseCFTimeOffset) -> datetime.timedelta:
    if False:
        i = 10
        return i + 15
    if isinstance(offset, datetime.timedelta):
        return offset
    elif isinstance(offset, (str, Tick)):
        return to_offset(offset).as_timedelta()
    else:
        raise ValueError

def _ceil_via_cftimeindex(date: CFTimeDatetime, freq: str | BaseCFTimeOffset):
    if False:
        i = 10
        return i + 15
    index = CFTimeIndex([date])
    return index.ceil(freq).item()