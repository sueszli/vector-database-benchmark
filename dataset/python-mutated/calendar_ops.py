from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import date_range_like, get_date_type
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.times import _should_cftime_be_used, convert_times
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
try:
    import cftime
except ImportError:
    cftime = None
_CALENDARS_WITHOUT_YEAR_ZERO = ['gregorian', 'proleptic_gregorian', 'julian', 'standard']

def _days_in_year(year, calendar, use_cftime=True):
    if False:
        for i in range(10):
            print('nop')
    'Return the number of days in the input year according to the input calendar.'
    date_type = get_date_type(calendar, use_cftime=use_cftime)
    if year == -1 and calendar in _CALENDARS_WITHOUT_YEAR_ZERO:
        difference = date_type(year + 2, 1, 1) - date_type(year, 1, 1)
    else:
        difference = date_type(year + 1, 1, 1) - date_type(year, 1, 1)
    return difference.days

def convert_calendar(obj, calendar, dim='time', align_on=None, missing=None, use_cftime=None):
    if False:
        i = 10
        return i + 15
    'Transform a time-indexed Dataset or DataArray to one that uses another calendar.\n\n    This function only converts the individual timestamps; it does not modify any\n    data except in dropping invalid/surplus dates, or inserting values for missing dates.\n\n    If the source and target calendars are both from a standard type, only the\n    type of the time array is modified. When converting to a calendar with a\n    leap year from to a calendar without a leap year, the 29th of February will\n    be removed from the array. In the other direction the 29th of February will\n    be missing in the output, unless `missing` is specified, in which case that\n    value is inserted. For conversions involving the `360_day` calendar, see Notes.\n\n    This method is safe to use with sub-daily data as it doesn\'t touch the time\n    part of the timestamps.\n\n    Parameters\n    ----------\n    obj : DataArray or Dataset\n      Input DataArray or Dataset with a time coordinate of a valid dtype\n      (:py:class:`numpy.datetime64`  or :py:class:`cftime.datetime`).\n    calendar : str\n      The target calendar name.\n    dim : str\n      Name of the time coordinate in the input DataArray or Dataset.\n    align_on : {None, \'date\', \'year\'}\n      Must be specified when either the source or target is a `"360_day"`\n      calendar; ignored otherwise. See Notes.\n    missing : any, optional\n      By default, i.e. if the value is None, this method will simply attempt\n      to convert the dates in the source calendar to the same dates in the\n      target calendar, and drop any of those that are not possible to\n      represent.  If a value is provided, a new time coordinate will be\n      created in the target calendar with the same frequency as the original\n      time coordinate; for any dates that are not present in the source, the\n      data will be filled with this value.  Note that using this mode requires\n      that the source data have an inferable frequency; for more information\n      see :py:func:`xarray.infer_freq`.  For certain frequency, source, and\n      target calendar combinations, this could result in many missing values, see notes.\n    use_cftime : bool, optional\n      Whether to use cftime objects in the output, only used if `calendar` is\n      one of {"proleptic_gregorian", "gregorian" or "standard"}.\n      If True, the new time axis uses cftime objects.\n      If None (default), it uses :py:class:`numpy.datetime64` values if the date\n          range permits it, and :py:class:`cftime.datetime` objects if not.\n      If False, it uses :py:class:`numpy.datetime64`  or fails.\n\n    Returns\n    -------\n      Copy of source with the time coordinate converted to the target calendar.\n      If `missing` was None (default), invalid dates in the new calendar are\n      dropped, but missing dates are not inserted.\n      If `missing` was given, the new data is reindexed to have a time axis\n      with the same frequency as the source, but in the new calendar; any\n      missing datapoints are filled with `missing`.\n\n    Notes\n    -----\n    Passing a value to `missing` is only usable if the source\'s time coordinate as an\n    inferable frequencies (see :py:func:`~xarray.infer_freq`) and is only appropriate\n    if the target coordinate, generated from this frequency, has dates equivalent to the\n    source. It is usually **not** appropriate to use this mode with:\n\n    - Period-end frequencies: \'A\', \'Y\', \'Q\' or \'M\', in opposition to \'AS\' \'YS\', \'QS\' and \'MS\'\n    - Sub-monthly frequencies that do not divide a day evenly: \'W\', \'nD\' where `n != 1`\n      or \'mH\' where 24 % m != 0).\n\n    If one of the source or target calendars is `"360_day"`, `align_on` must\n    be specified and two options are offered.\n\n    "year"\n      The dates are translated according to their relative position in the year,\n      ignoring their original month and day information, meaning that the\n      missing/surplus days are added/removed at regular intervals.\n\n      From a `360_day` to a standard calendar, the output will be missing the\n      following dates (day of year in parentheses):\n        To a leap year:\n          January 31st (31), March 31st (91), June 1st (153), July 31st (213),\n          September 31st (275) and November 30th (335).\n        To a non-leap year:\n          February 6th (36), April 19th (109), July 2nd (183),\n          September 12th (255), November 25th (329).\n\n      From a standard calendar to a `"360_day"`, the following dates in the\n      source array will be dropped:\n        From a leap year:\n          January 31st (31), April 1st (92), June 1st (153), August 1st (214),\n          September 31st (275), December 1st (336)\n        From a non-leap year:\n          February 6th (37), April 20th (110), July 2nd (183),\n          September 13th (256), November 25th (329)\n\n      This option is best used on daily and subdaily data.\n\n    "date"\n      The month/day information is conserved and invalid dates are dropped\n      from the output. This means that when converting from a `"360_day"` to a\n      standard calendar, all 31sts (Jan, March, May, July, August, October and\n      December) will be missing as there is no equivalent dates in the\n      `"360_day"` calendar and the 29th (on non-leap years) and 30th of February\n      will be dropped as there are no equivalent dates in a standard calendar.\n\n      This option is best used with data on a frequency coarser than daily.\n    '
    from xarray.core.dataarray import DataArray
    time = obj[dim]
    if not _contains_datetime_like_objects(time.variable):
        raise ValueError(f'Coordinate {dim} must contain datetime objects.')
    use_cftime = _should_cftime_be_used(time, calendar, use_cftime)
    source_calendar = time.dt.calendar
    if source_calendar == calendar and is_np_datetime_like(time.dtype) ^ use_cftime:
        return obj
    if (time.dt.year == 0).any() and calendar in _CALENDARS_WITHOUT_YEAR_ZERO:
        raise ValueError(f'Source time coordinate contains dates with year 0, which is not supported by target calendar {calendar}.')
    if (source_calendar == '360_day' or calendar == '360_day') and align_on is None:
        raise ValueError("Argument `align_on` must be specified with either 'date' or 'year' when converting to or from a '360_day' calendar.")
    if source_calendar != '360_day' and calendar != '360_day':
        align_on = 'date'
    out = obj.copy()
    if align_on == 'year':
        new_doy = time.groupby(f'{dim}.year').map(_interpolate_day_of_year, target_calendar=calendar, use_cftime=use_cftime)
        out[dim] = DataArray([_convert_to_new_calendar_with_new_day_of_year(date, newdoy, calendar, use_cftime) for (date, newdoy) in zip(time.variable._data.array, new_doy)], dims=(dim,), name=dim)
        out = out.isel({dim: np.unique(out[dim], return_index=True)[1]})
    elif align_on == 'date':
        new_times = convert_times(time.data, get_date_type(calendar, use_cftime=use_cftime), raise_on_invalid=False)
        out[dim] = new_times
        out = out.where(out[dim].notnull(), drop=True)
    if missing is not None:
        time_target = date_range_like(time, calendar=calendar, use_cftime=use_cftime)
        out = out.reindex({dim: time_target}, fill_value=missing)
    out[dim].attrs.update(time.attrs)
    out[dim].attrs.pop('calendar', None)
    return out

def _interpolate_day_of_year(time, target_calendar, use_cftime):
    if False:
        for i in range(10):
            print('nop')
    'Returns the nearest day in the target calendar of the corresponding\n    "decimal year" in the source calendar.\n    '
    year = int(time.dt.year[0])
    source_calendar = time.dt.calendar
    return np.round(_days_in_year(year, target_calendar, use_cftime) * time.dt.dayofyear / _days_in_year(year, source_calendar, use_cftime)).astype(int)

def _convert_to_new_calendar_with_new_day_of_year(date, day_of_year, calendar, use_cftime):
    if False:
        print('Hello World!')
    "Convert a datetime object to another calendar with a new day of year.\n\n    Redefines the day of year (and thus ignores the month and day information\n    from the source datetime).\n    Nanosecond information is lost as cftime.datetime doesn't support it.\n    "
    new_date = cftime.num2date(day_of_year - 1, f'days since {date.year}-01-01', calendar=calendar if use_cftime else 'standard')
    try:
        return get_date_type(calendar, use_cftime)(date.year, new_date.month, new_date.day, date.hour, date.minute, date.second, date.microsecond)
    except ValueError:
        return np.nan

def _datetime_to_decimal_year(times, dim='time', calendar=None):
    if False:
        i = 10
        return i + 15
    'Convert a datetime DataArray to decimal years according to its calendar or the given one.\n\n    The decimal year of a timestamp is its year plus its sub-year component\n    converted to the fraction of its year.\n    Ex: \'2000-03-01 12:00\' is 2000.1653 in a standard calendar,\n      2000.16301 in a "noleap" or 2000.16806 in a "360_day".\n    '
    from xarray.core.dataarray import DataArray
    calendar = calendar or times.dt.calendar
    if is_np_datetime_like(times.dtype):
        times = times.copy(data=convert_times(times.values, get_date_type('standard')))

    def _make_index(time):
        if False:
            print('Hello World!')
        year = int(time.dt.year[0])
        doys = cftime.date2num(time, f'days since {year:04d}-01-01', calendar=calendar)
        return DataArray(year + doys / _days_in_year(year, calendar), dims=(dim,), coords=time.coords, name=dim)
    return times.groupby(f'{dim}.year').map(_make_index)

def interp_calendar(source, target, dim='time'):
    if False:
        i = 10
        return i + 15
    'Interpolates a DataArray or Dataset indexed by a time coordinate to\n    another calendar based on decimal year measure.\n\n    Each timestamp in `source` and `target` are first converted to their decimal\n    year equivalent then `source` is interpolated on the target coordinate.\n    The decimal year of a timestamp is its year plus its sub-year component\n    converted to the fraction of its year. For example "2000-03-01 12:00" is\n    2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.\n\n    This method should only be used when the time (HH:MM:SS) information of\n    time coordinate is not important.\n\n    Parameters\n    ----------\n    source: DataArray or Dataset\n      The source data to interpolate; must have a time coordinate of a valid\n      dtype (:py:class:`numpy.datetime64` or :py:class:`cftime.datetime` objects)\n    target: DataArray, DatetimeIndex, or CFTimeIndex\n      The target time coordinate of a valid dtype (np.datetime64 or cftime objects)\n    dim : str\n      The time coordinate name.\n\n    Return\n    ------\n    DataArray or Dataset\n      The source interpolated on the decimal years of target,\n    '
    from xarray.core.dataarray import DataArray
    if isinstance(target, (pd.DatetimeIndex, CFTimeIndex)):
        target = DataArray(target, dims=(dim,), name=dim)
    if not _contains_datetime_like_objects(source[dim].variable) or not _contains_datetime_like_objects(target.variable):
        raise ValueError(f"Both 'source.{dim}' and 'target' must contain datetime objects.")
    source_calendar = source[dim].dt.calendar
    target_calendar = target.dt.calendar
    if (source[dim].time.dt.year == 0).any() and target_calendar in _CALENDARS_WITHOUT_YEAR_ZERO:
        raise ValueError(f'Source time coordinate contains dates with year 0, which is not supported by target calendar {target_calendar}.')
    out = source.copy()
    out[dim] = _datetime_to_decimal_year(source[dim], dim=dim, calendar=source_calendar)
    target_idx = _datetime_to_decimal_year(target, dim=dim, calendar=target_calendar)
    out = out.interp(**{dim: target_idx})
    out[dim] = target
    return out