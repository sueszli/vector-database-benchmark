"""Time offset classes for use with cftime.datetime objects"""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import _is_standard_calendar, _should_cftime_be_used, convert_time_or_go_back, format_cftime_datetime
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import NoDefault, count_not_none, nanosecond_precision_timestamp, no_default
from xarray.core.utils import emit_user_level_warning
try:
    import cftime
except ImportError:
    cftime = None
if TYPE_CHECKING:
    from xarray.core.types import InclusiveOptions, SideOptions

def get_date_type(calendar, use_cftime=True):
    if False:
        print('Hello World!')
    'Return the cftime date type for a given calendar name.'
    if cftime is None:
        raise ImportError('cftime is required for dates with non-standard calendars')
    else:
        if _is_standard_calendar(calendar) and (not use_cftime):
            return pd.Timestamp
        calendars = {'noleap': cftime.DatetimeNoLeap, '360_day': cftime.Datetime360Day, '365_day': cftime.DatetimeNoLeap, '366_day': cftime.DatetimeAllLeap, 'gregorian': cftime.DatetimeGregorian, 'proleptic_gregorian': cftime.DatetimeProlepticGregorian, 'julian': cftime.DatetimeJulian, 'all_leap': cftime.DatetimeAllLeap, 'standard': cftime.DatetimeGregorian}
        return calendars[calendar]

class BaseCFTimeOffset:
    _freq: ClassVar[str | None] = None
    _day_option: ClassVar[str | None] = None

    def __init__(self, n: int=1):
        if False:
            i = 10
            return i + 15
        if not isinstance(n, int):
            raise TypeError(f"The provided multiple 'n' must be an integer. Instead a value of type {type(n)!r} was provided.")
        self.n = n

    def rule_code(self):
        if False:
            return 10
        return self._freq

    def __eq__(self, other):
        if False:
            return 10
        return self.n == other.n and self.rule_code() == other.rule_code()

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __add__(self, other):
        if False:
            return 10
        return self.__apply__(other)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract a cftime.datetime from a time offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            return NotImplemented

    def __mul__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(n=other * self.n)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self * -1

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__mul__(other)

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__add__(other)

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, BaseCFTimeOffset) and type(self) != type(other):
            raise TypeError('Cannot subtract cftime offsets of differing types')
        return -self + other

    def __apply__(self):
        if False:
            print('Hello World!')
        return NotImplemented

    def onOffset(self, date):
        if False:
            for i in range(10):
                print('nop')
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        test_date = self + date - self
        return date == test_date

    def rollforward(self, date):
        if False:
            i = 10
            return i + 15
        if self.onOffset(date):
            return date
        else:
            return date + type(self)()

    def rollback(self, date):
        if False:
            print('Hello World!')
        if self.onOffset(date):
            return date
        else:
            return date - type(self)()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'<{type(self).__name__}: n={self.n}>'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self)

    def _get_offset_day(self, other):
        if False:
            return 10
        return _get_day_of_month(other, self._day_option)

class Tick(BaseCFTimeOffset):

    def _next_higher_resolution(self):
        if False:
            i = 10
            return i + 15
        self_type = type(self)
        if self_type not in [Day, Hour, Minute, Second, Millisecond]:
            raise ValueError('Could not convert to integer offset at any resolution')
        if type(self) is Day:
            return Hour(self.n * 24)
        if type(self) is Hour:
            return Minute(self.n * 60)
        if type(self) is Minute:
            return Second(self.n * 60)
        if type(self) is Second:
            return Millisecond(self.n * 1000)
        if type(self) is Millisecond:
            return Microsecond(self.n * 1000)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, (int, float)):
            return NotImplemented
        if isinstance(other, float):
            n = other * self.n
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return type(self)(n=other * self.n)

    def as_timedelta(self):
        if False:
            print('Hello World!')
        'All Tick subclasses must implement an as_timedelta method.'
        raise NotImplementedError

def _get_day_of_month(other, day_option):
    if False:
        while True:
            i = 10
    "Find the day in `other`'s month that satisfies a BaseCFTimeOffset's\n    onOffset policy, as described by the `day_option` argument.\n\n    Parameters\n    ----------\n    other : cftime.datetime\n    day_option : 'start', 'end'\n        'start': returns 1\n        'end': returns last day of the month\n\n    Returns\n    -------\n    day_of_month : int\n\n    "
    if day_option == 'start':
        return 1
    elif day_option == 'end':
        return _days_in_month(other)
    elif day_option is None:
        raise NotImplementedError()
    else:
        raise ValueError(day_option)

def _days_in_month(date):
    if False:
        i = 10
        return i + 15
    'The number of days in the month of the given date'
    if date.month == 12:
        reference = type(date)(date.year + 1, 1, 1)
    else:
        reference = type(date)(date.year, date.month + 1, 1)
    return (reference - timedelta(days=1)).day

def _adjust_n_months(other_day, n, reference_day):
    if False:
        return 10
    'Adjust the number of times a monthly offset is applied based\n    on the day of a given date, and the reference day provided.\n    '
    if n > 0 and other_day < reference_day:
        n = n - 1
    elif n <= 0 and other_day > reference_day:
        n = n + 1
    return n

def _adjust_n_years(other, n, month, reference_day):
    if False:
        print('Hello World!')
    'Adjust the number of times an annual offset is applied based on\n    another date, and the reference day provided'
    if n > 0:
        if other.month < month or (other.month == month and other.day < reference_day):
            n -= 1
    elif other.month > month or (other.month == month and other.day > reference_day):
        n += 1
    return n

def _shift_month(date, months, day_option='start'):
    if False:
        for i in range(10):
            print('nop')
    'Shift the date to a month start or end a given number of months away.'
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    delta_year = (date.month + months) // 12
    month = (date.month + months) % 12
    if month == 0:
        month = 12
        delta_year = delta_year - 1
    year = date.year + delta_year
    if day_option == 'start':
        day = 1
    elif day_option == 'end':
        reference = type(date)(year, month, 1)
        day = _days_in_month(reference)
    else:
        raise ValueError(day_option)
    return date.replace(year=year, month=month, day=day)

def roll_qtrday(other, n, month, day_option, modby=3):
    if False:
        i = 10
        return i + 15
    "Possibly increment or decrement the number of periods to shift\n    based on rollforward/rollbackward conventions.\n\n    Parameters\n    ----------\n    other : cftime.datetime\n    n : number of periods to increment, before adjusting for rolling\n    month : int reference month giving the first month of the year\n    day_option : 'start', 'end'\n        The convention to use in finding the day in a given month against\n        which to compare for rollforward/rollbackward decisions.\n    modby : int 3 for quarters, 12 for years\n\n    Returns\n    -------\n    n : int number of periods to increment\n\n    See Also\n    --------\n    _get_day_of_month : Find the day in a month provided an offset.\n    "
    months_since = other.month % modby - month % modby
    if n > 0:
        if months_since < 0 or (months_since == 0 and other.day < _get_day_of_month(other, day_option)):
            n -= 1
    elif months_since > 0 or (months_since == 0 and other.day > _get_day_of_month(other, day_option)):
        n += 1
    return n

def _validate_month(month, default_month):
    if False:
        while True:
            i = 10
    result_month = default_month if month is None else month
    if not isinstance(result_month, int):
        raise TypeError(f"'self.month' must be an integer value between 1 and 12.  Instead, it was set to a value of {result_month!r}")
    elif not 1 <= result_month <= 12:
        raise ValueError(f"'self.month' must be an integer value between 1 and 12.  Instead, it was set to a value of {result_month!r}")
    return result_month

class MonthBegin(BaseCFTimeOffset):
    _freq = 'MS'

    def __apply__(self, other):
        if False:
            for i in range(10):
                print('nop')
        n = _adjust_n_months(other.day, self.n, 1)
        return _shift_month(other, n, 'start')

    def onOffset(self, date):
        if False:
            while True:
                i = 10
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        return date.day == 1

class MonthEnd(BaseCFTimeOffset):
    _freq = 'M'

    def __apply__(self, other):
        if False:
            i = 10
            return i + 15
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_month(other, n, 'end')

    def onOffset(self, date):
        if False:
            i = 10
            return i + 15
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        return date.day == _days_in_month(date)
_MONTH_ABBREVIATIONS = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

class QuarterOffset(BaseCFTimeOffset):
    """Quarter representation copied off of pandas/tseries/offsets.py"""
    _freq: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        if False:
            i = 10
            return i + 15
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        if False:
            while True:
                i = 10
        months_since = other.month % 3 - self.month % 3
        qtrs = roll_qtrday(other, self.n, self.month, day_option=self._day_option, modby=3)
        months = qtrs * 3 - months_since
        return _shift_month(other, months, self._day_option)

    def onOffset(self, date):
        if False:
            print('Hello World!')
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        mod_month = (date.month - self.month) % 3
        return mod_month == 0 and date.day == self._get_offset_day(date)

    def __sub__(self, other):
        if False:
            return 10
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if False:
            return 10
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        if False:
            i = 10
            return i + 15
        return f'{self._freq}-{_MONTH_ABBREVIATIONS[self.month]}'

    def __str__(self):
        if False:
            return 10
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class QuarterBegin(QuarterOffset):
    _default_month = 3
    _freq = 'QS'
    _day_option = 'start'

    def rollforward(self, date):
        if False:
            print('Hello World!')
        'Roll date forward to nearest start of quarter'
        if self.onOffset(date):
            return date
        else:
            return date + QuarterBegin(month=self.month)

    def rollback(self, date):
        if False:
            print('Hello World!')
        'Roll date backward to nearest start of quarter'
        if self.onOffset(date):
            return date
        else:
            return date - QuarterBegin(month=self.month)

class QuarterEnd(QuarterOffset):
    _default_month = 3
    _freq = 'Q'
    _day_option = 'end'

    def rollforward(self, date):
        if False:
            print('Hello World!')
        'Roll date forward to nearest end of quarter'
        if self.onOffset(date):
            return date
        else:
            return date + QuarterEnd(month=self.month)

    def rollback(self, date):
        if False:
            print('Hello World!')
        'Roll date backward to nearest end of quarter'
        if self.onOffset(date):
            return date
        else:
            return date - QuarterEnd(month=self.month)

class YearOffset(BaseCFTimeOffset):
    _freq: ClassVar[str]
    _day_option: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        if False:
            for i in range(10):
                print('nop')
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        if False:
            while True:
                i = 10
        reference_day = _get_day_of_month(other, self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_month(other, months, self._day_option)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        if False:
            print('Hello World!')
        return f'{self._freq}-{_MONTH_ABBREVIATIONS[self.month]}'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class YearBegin(YearOffset):
    _freq = 'AS'
    _day_option = 'start'
    _default_month = 1

    def onOffset(self, date):
        if False:
            while True:
                i = 10
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        return date.day == 1 and date.month == self.month

    def rollforward(self, date):
        if False:
            for i in range(10):
                print('nop')
        'Roll date forward to nearest start of year'
        if self.onOffset(date):
            return date
        else:
            return date + YearBegin(month=self.month)

    def rollback(self, date):
        if False:
            return 10
        'Roll date backward to nearest start of year'
        if self.onOffset(date):
            return date
        else:
            return date - YearBegin(month=self.month)

class YearEnd(YearOffset):
    _freq = 'A'
    _day_option = 'end'
    _default_month = 12

    def onOffset(self, date):
        if False:
            print('Hello World!')
        'Check if the given date is in the set of possible dates created\n        using a length-one version of this offset class.'
        return date.day == _days_in_month(date) and date.month == self.month

    def rollforward(self, date):
        if False:
            for i in range(10):
                print('nop')
        'Roll date forward to nearest end of year'
        if self.onOffset(date):
            return date
        else:
            return date + YearEnd(month=self.month)

    def rollback(self, date):
        if False:
            return 10
        'Roll date backward to nearest end of year'
        if self.onOffset(date):
            return date
        else:
            return date - YearEnd(month=self.month)

class Day(Tick):
    _freq = 'D'

    def as_timedelta(self):
        if False:
            return 10
        return timedelta(days=self.n)

    def __apply__(self, other):
        if False:
            while True:
                i = 10
        return other + self.as_timedelta()

class Hour(Tick):
    _freq = 'H'

    def as_timedelta(self):
        if False:
            i = 10
            return i + 15
        return timedelta(hours=self.n)

    def __apply__(self, other):
        if False:
            i = 10
            return i + 15
        return other + self.as_timedelta()

class Minute(Tick):
    _freq = 'T'

    def as_timedelta(self):
        if False:
            print('Hello World!')
        return timedelta(minutes=self.n)

    def __apply__(self, other):
        if False:
            while True:
                i = 10
        return other + self.as_timedelta()

class Second(Tick):
    _freq = 'S'

    def as_timedelta(self):
        if False:
            i = 10
            return i + 15
        return timedelta(seconds=self.n)

    def __apply__(self, other):
        if False:
            return 10
        return other + self.as_timedelta()

class Millisecond(Tick):
    _freq = 'L'

    def as_timedelta(self):
        if False:
            while True:
                i = 10
        return timedelta(milliseconds=self.n)

    def __apply__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other + self.as_timedelta()

class Microsecond(Tick):
    _freq = 'U'

    def as_timedelta(self):
        if False:
            print('Hello World!')
        return timedelta(microseconds=self.n)

    def __apply__(self, other):
        if False:
            return 10
        return other + self.as_timedelta()
_FREQUENCIES = {'A': YearEnd, 'AS': YearBegin, 'Y': YearEnd, 'YS': YearBegin, 'Q': partial(QuarterEnd, month=12), 'QS': partial(QuarterBegin, month=1), 'M': MonthEnd, 'MS': MonthBegin, 'D': Day, 'H': Hour, 'T': Minute, 'min': Minute, 'S': Second, 'L': Millisecond, 'ms': Millisecond, 'U': Microsecond, 'us': Microsecond, 'AS-JAN': partial(YearBegin, month=1), 'AS-FEB': partial(YearBegin, month=2), 'AS-MAR': partial(YearBegin, month=3), 'AS-APR': partial(YearBegin, month=4), 'AS-MAY': partial(YearBegin, month=5), 'AS-JUN': partial(YearBegin, month=6), 'AS-JUL': partial(YearBegin, month=7), 'AS-AUG': partial(YearBegin, month=8), 'AS-SEP': partial(YearBegin, month=9), 'AS-OCT': partial(YearBegin, month=10), 'AS-NOV': partial(YearBegin, month=11), 'AS-DEC': partial(YearBegin, month=12), 'A-JAN': partial(YearEnd, month=1), 'A-FEB': partial(YearEnd, month=2), 'A-MAR': partial(YearEnd, month=3), 'A-APR': partial(YearEnd, month=4), 'A-MAY': partial(YearEnd, month=5), 'A-JUN': partial(YearEnd, month=6), 'A-JUL': partial(YearEnd, month=7), 'A-AUG': partial(YearEnd, month=8), 'A-SEP': partial(YearEnd, month=9), 'A-OCT': partial(YearEnd, month=10), 'A-NOV': partial(YearEnd, month=11), 'A-DEC': partial(YearEnd, month=12), 'QS-JAN': partial(QuarterBegin, month=1), 'QS-FEB': partial(QuarterBegin, month=2), 'QS-MAR': partial(QuarterBegin, month=3), 'QS-APR': partial(QuarterBegin, month=4), 'QS-MAY': partial(QuarterBegin, month=5), 'QS-JUN': partial(QuarterBegin, month=6), 'QS-JUL': partial(QuarterBegin, month=7), 'QS-AUG': partial(QuarterBegin, month=8), 'QS-SEP': partial(QuarterBegin, month=9), 'QS-OCT': partial(QuarterBegin, month=10), 'QS-NOV': partial(QuarterBegin, month=11), 'QS-DEC': partial(QuarterBegin, month=12), 'Q-JAN': partial(QuarterEnd, month=1), 'Q-FEB': partial(QuarterEnd, month=2), 'Q-MAR': partial(QuarterEnd, month=3), 'Q-APR': partial(QuarterEnd, month=4), 'Q-MAY': partial(QuarterEnd, month=5), 'Q-JUN': partial(QuarterEnd, month=6), 'Q-JUL': partial(QuarterEnd, month=7), 'Q-AUG': partial(QuarterEnd, month=8), 'Q-SEP': partial(QuarterEnd, month=9), 'Q-OCT': partial(QuarterEnd, month=10), 'Q-NOV': partial(QuarterEnd, month=11), 'Q-DEC': partial(QuarterEnd, month=12)}
_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = f'^((?P<multiple>\\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$'
CFTIME_TICKS = (Day, Hour, Minute, Second)

def to_offset(freq):
    if False:
        i = 10
        return i + 15
    'Convert a frequency string to the appropriate subclass of\n    BaseCFTimeOffset.'
    if isinstance(freq, BaseCFTimeOffset):
        return freq
    else:
        try:
            freq_data = re.match(_PATTERN, freq).groupdict()
        except AttributeError:
            raise ValueError('Invalid frequency string provided')
    freq = freq_data['freq']
    multiples = freq_data['multiple']
    multiples = 1 if multiples is None else int(multiples)
    return _FREQUENCIES[freq](n=multiples)

def to_cftime_datetime(date_str_or_date, calendar=None):
    if False:
        print('Hello World!')
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if isinstance(date_str_or_date, str):
        if calendar is None:
            raise ValueError('If converting a string to a cftime.datetime object, a calendar type must be provided')
        (date, _) = _parse_iso8601_with_reso(get_date_type(calendar), date_str_or_date)
        return date
    elif isinstance(date_str_or_date, cftime.datetime):
        return date_str_or_date
    elif isinstance(date_str_or_date, (datetime, pd.Timestamp)):
        return cftime.DatetimeProlepticGregorian(*date_str_or_date.timetuple())
    else:
        raise TypeError(f'date_str_or_date must be a string or a subclass of cftime.datetime. Instead got {date_str_or_date!r}.')

def normalize_date(date):
    if False:
        return 10
    'Round datetime down to midnight.'
    return date.replace(hour=0, minute=0, second=0, microsecond=0)

def _maybe_normalize_date(date, normalize):
    if False:
        for i in range(10):
            print('nop')
    'Round datetime down to midnight if normalize is True.'
    if normalize:
        return normalize_date(date)
    else:
        return date

def _generate_linear_range(start, end, periods):
    if False:
        while True:
            i = 10
    'Generate an equally-spaced sequence of cftime.datetime objects between\n    and including two dates (whose length equals the number of periods).'
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    total_seconds = (end - start).total_seconds()
    values = np.linspace(0.0, total_seconds, periods, endpoint=True)
    units = f'seconds since {format_cftime_datetime(start)}'
    calendar = start.calendar
    return cftime.num2date(values, units=units, calendar=calendar, only_use_cftime_datetimes=True)

def _generate_range(start, end, periods, offset):
    if False:
        for i in range(10):
            print('nop')
    'Generate a regular range of cftime.datetime objects with a\n    given time offset.\n\n    Adapted from pandas.tseries.offsets.generate_range.\n\n    Parameters\n    ----------\n    start : cftime.datetime, or None\n        Start of range\n    end : cftime.datetime, or None\n        End of range\n    periods : int, or None\n        Number of elements in the sequence\n    offset : BaseCFTimeOffset\n        An offset class designed for working with cftime.datetime objects\n\n    Returns\n    -------\n    A generator object\n    '
    if start:
        start = offset.rollforward(start)
    if end:
        end = offset.rollback(end)
    if periods is None and end < start:
        end = None
        periods = 0
    if end is None:
        end = start + (periods - 1) * offset
    if start is None:
        start = end - (periods - 1) * offset
    current = start
    if offset.n >= 0:
        while current <= end:
            yield current
            next_date = current + offset
            if next_date <= current:
                raise ValueError(f'Offset {offset} did not increment date')
            current = next_date
    else:
        while current >= end:
            yield current
            next_date = current + offset
            if next_date >= current:
                raise ValueError(f'Offset {offset} did not decrement date')
            current = next_date

def _translate_closed_to_inclusive(closed):
    if False:
        i = 10
        return i + 15
    'Follows code added in pandas #43504.'
    emit_user_level_warning('Following pandas, the `closed` parameter is deprecated in favor of the `inclusive` parameter, and will be removed in a future version of xarray.', FutureWarning)
    if closed is None:
        inclusive = 'both'
    elif closed in ('left', 'right'):
        inclusive = closed
    else:
        raise ValueError(f"Argument `closed` must be either 'left', 'right', or None. Got {closed!r}.")
    return inclusive

def _infer_inclusive(closed, inclusive):
    if False:
        while True:
            i = 10
    'Follows code added in pandas #43504.'
    if closed is not no_default and inclusive is not None:
        raise ValueError('Following pandas, deprecated argument `closed` cannot be passed if argument `inclusive` is not None.')
    if closed is not no_default:
        inclusive = _translate_closed_to_inclusive(closed)
    elif inclusive is None:
        inclusive = 'both'
    return inclusive

def cftime_range(start=None, end=None, periods=None, freq='D', normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard'):
    if False:
        while True:
            i = 10
    'Return a fixed frequency CFTimeIndex.\n\n    Parameters\n    ----------\n    start : str or cftime.datetime, optional\n        Left bound for generating dates.\n    end : str or cftime.datetime, optional\n        Right bound for generating dates.\n    periods : int, optional\n        Number of periods to generate.\n    freq : str or None, default: "D"\n        Frequency strings can have multiples, e.g. "5H".\n    normalize : bool, default: False\n        Normalize start/end dates to midnight before generating date range.\n    name : str, default: None\n        Name of the resulting index\n    closed : {None, "left", "right"}, default: "NO_DEFAULT"\n        Make the interval closed with respect to the given frequency to the\n        "left", "right", or both sides (None).\n\n        .. deprecated:: 2023.02.0\n            Following pandas, the ``closed`` parameter is deprecated in favor\n            of the ``inclusive`` parameter, and will be removed in a future\n            version of xarray.\n\n    inclusive : {None, "both", "neither", "left", "right"}, default None\n        Include boundaries; whether to set each bound as closed or open.\n\n        .. versionadded:: 2023.02.0\n\n    calendar : str, default: "standard"\n        Calendar type for the datetimes.\n\n    Returns\n    -------\n    CFTimeIndex\n\n    Notes\n    -----\n    This function is an analog of ``pandas.date_range`` for use in generating\n    sequences of ``cftime.datetime`` objects.  It supports most of the\n    features of ``pandas.date_range`` (e.g. specifying how the index is\n    ``closed`` on either side, or whether or not to ``normalize`` the start and\n    end bounds); however, there are some notable exceptions:\n\n    - You cannot specify a ``tz`` (time zone) argument.\n    - Start or end dates specified as partial-datetime strings must use the\n      `ISO-8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.\n    - It supports many, but not all, frequencies supported by\n      ``pandas.date_range``.  For example it does not currently support any of\n      the business-related or semi-monthly frequencies.\n    - Compound sub-monthly frequencies are not supported, e.g. \'1H1min\', as\n      these can easily be written in terms of the finest common resolution,\n      e.g. \'61min\'.\n\n    Valid simple frequency strings for use with ``cftime``-calendars include\n    any multiples of the following.\n\n    +--------+--------------------------+\n    | Alias  | Description              |\n    +========+==========================+\n    | A, Y   | Year-end frequency       |\n    +--------+--------------------------+\n    | AS, YS | Year-start frequency     |\n    +--------+--------------------------+\n    | Q      | Quarter-end frequency    |\n    +--------+--------------------------+\n    | QS     | Quarter-start frequency  |\n    +--------+--------------------------+\n    | M      | Month-end frequency      |\n    +--------+--------------------------+\n    | MS     | Month-start frequency    |\n    +--------+--------------------------+\n    | D      | Day frequency            |\n    +--------+--------------------------+\n    | H      | Hour frequency           |\n    +--------+--------------------------+\n    | T, min | Minute frequency         |\n    +--------+--------------------------+\n    | S      | Second frequency         |\n    +--------+--------------------------+\n    | L, ms  | Millisecond frequency    |\n    +--------+--------------------------+\n    | U, us  | Microsecond frequency    |\n    +--------+--------------------------+\n\n    Any multiples of the following anchored offsets are also supported.\n\n    +----------+--------------------------------------------------------------------+\n    | Alias    | Description                                                        |\n    +==========+====================================================================+\n    | A(S)-JAN | Annual frequency, anchored at the end (or beginning) of January    |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-FEB | Annual frequency, anchored at the end (or beginning) of February   |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-MAR | Annual frequency, anchored at the end (or beginning) of March      |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-APR | Annual frequency, anchored at the end (or beginning) of April      |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-MAY | Annual frequency, anchored at the end (or beginning) of May        |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-JUN | Annual frequency, anchored at the end (or beginning) of June       |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-JUL | Annual frequency, anchored at the end (or beginning) of July       |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-AUG | Annual frequency, anchored at the end (or beginning) of August     |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-SEP | Annual frequency, anchored at the end (or beginning) of September  |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-OCT | Annual frequency, anchored at the end (or beginning) of October    |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-NOV | Annual frequency, anchored at the end (or beginning) of November   |\n    +----------+--------------------------------------------------------------------+\n    | A(S)-DEC | Annual frequency, anchored at the end (or beginning) of December   |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-JAN | Quarter frequency, anchored at the end (or beginning) of January   |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-FEB | Quarter frequency, anchored at the end (or beginning) of February  |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-MAR | Quarter frequency, anchored at the end (or beginning) of March     |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-APR | Quarter frequency, anchored at the end (or beginning) of April     |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-MAY | Quarter frequency, anchored at the end (or beginning) of May       |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-JUN | Quarter frequency, anchored at the end (or beginning) of June      |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-JUL | Quarter frequency, anchored at the end (or beginning) of July      |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-AUG | Quarter frequency, anchored at the end (or beginning) of August    |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-SEP | Quarter frequency, anchored at the end (or beginning) of September |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-OCT | Quarter frequency, anchored at the end (or beginning) of October   |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-NOV | Quarter frequency, anchored at the end (or beginning) of November  |\n    +----------+--------------------------------------------------------------------+\n    | Q(S)-DEC | Quarter frequency, anchored at the end (or beginning) of December  |\n    +----------+--------------------------------------------------------------------+\n\n    Finally, the following calendar aliases are supported.\n\n    +--------------------------------+---------------------------------------+\n    | Alias                          | Date type                             |\n    +================================+=======================================+\n    | standard, gregorian            | ``cftime.DatetimeGregorian``          |\n    +--------------------------------+---------------------------------------+\n    | proleptic_gregorian            | ``cftime.DatetimeProlepticGregorian`` |\n    +--------------------------------+---------------------------------------+\n    | noleap, 365_day                | ``cftime.DatetimeNoLeap``             |\n    +--------------------------------+---------------------------------------+\n    | all_leap, 366_day              | ``cftime.DatetimeAllLeap``            |\n    +--------------------------------+---------------------------------------+\n    | 360_day                        | ``cftime.Datetime360Day``             |\n    +--------------------------------+---------------------------------------+\n    | julian                         | ``cftime.DatetimeJulian``             |\n    +--------------------------------+---------------------------------------+\n\n    Examples\n    --------\n    This function returns a ``CFTimeIndex``, populated with ``cftime.datetime``\n    objects associated with the specified calendar type, e.g.\n\n    >>> xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")\n    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,\n                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],\n                dtype=\'object\', length=6, calendar=\'noleap\', freq=\'2MS\')\n\n    As in the standard pandas function, three of the ``start``, ``end``,\n    ``periods``, or ``freq`` arguments must be specified at a given time, with\n    the other set to ``None``.  See the `pandas documentation\n    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_\n    for more examples of the behavior of ``date_range`` with each of the\n    parameters.\n\n    See Also\n    --------\n    pandas.date_range\n    '
    if count_not_none(start, end, periods, freq) != 3:
        raise ValueError("Of the arguments 'start', 'end', 'periods', and 'freq', three must be specified at a time.")
    if start is not None:
        start = to_cftime_datetime(start, calendar)
        start = _maybe_normalize_date(start, normalize)
    if end is not None:
        end = to_cftime_datetime(end, calendar)
        end = _maybe_normalize_date(end, normalize)
    if freq is None:
        dates = _generate_linear_range(start, end, periods)
    else:
        offset = to_offset(freq)
        dates = np.array(list(_generate_range(start, end, periods, offset)))
    inclusive = _infer_inclusive(closed, inclusive)
    if inclusive == 'neither':
        left_closed = False
        right_closed = False
    elif inclusive == 'left':
        left_closed = True
        right_closed = False
    elif inclusive == 'right':
        left_closed = False
        right_closed = True
    elif inclusive == 'both':
        left_closed = True
        right_closed = True
    else:
        raise ValueError(f"Argument `inclusive` must be either 'both', 'neither', 'left', 'right', or None.  Got {inclusive}.")
    if not left_closed and len(dates) and (start is not None) and (dates[0] == start):
        dates = dates[1:]
    if not right_closed and len(dates) and (end is not None) and (dates[-1] == end):
        dates = dates[:-1]
    return CFTimeIndex(dates, name=name)

def date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard', use_cftime=None):
    if False:
        return 10
    'Return a fixed frequency datetime index.\n\n    The type (:py:class:`xarray.CFTimeIndex` or :py:class:`pandas.DatetimeIndex`)\n    of the returned index depends on the requested calendar and on `use_cftime`.\n\n    Parameters\n    ----------\n    start : str or datetime-like, optional\n        Left bound for generating dates.\n    end : str or datetime-like, optional\n        Right bound for generating dates.\n    periods : int, optional\n        Number of periods to generate.\n    freq : str or None, default: "D"\n        Frequency strings can have multiples, e.g. "5H".\n    tz : str or tzinfo, optional\n        Time zone name for returning localized DatetimeIndex, for example\n        \'Asia/Hong_Kong\'. By default, the resulting DatetimeIndex is\n        timezone-naive. Only valid with pandas DatetimeIndex.\n    normalize : bool, default: False\n        Normalize start/end dates to midnight before generating date range.\n    name : str, default: None\n        Name of the resulting index\n    closed : {None, "left", "right"}, default: "NO_DEFAULT"\n        Make the interval closed with respect to the given frequency to the\n        "left", "right", or both sides (None).\n\n        .. deprecated:: 2023.02.0\n            Following pandas, the `closed` parameter is deprecated in favor\n            of the `inclusive` parameter, and will be removed in a future\n            version of xarray.\n\n    inclusive : {None, "both", "neither", "left", "right"}, default: None\n        Include boundaries; whether to set each bound as closed or open.\n\n        .. versionadded:: 2023.02.0\n\n    calendar : str, default: "standard"\n        Calendar type for the datetimes.\n    use_cftime : boolean, optional\n        If True, always return a CFTimeIndex.\n        If False, return a pd.DatetimeIndex if possible or raise a ValueError.\n        If None (default), return a pd.DatetimeIndex if possible,\n        otherwise return a CFTimeIndex. Defaults to False if `tz` is not None.\n\n    Returns\n    -------\n    CFTimeIndex or pd.DatetimeIndex\n\n    See also\n    --------\n    pandas.date_range\n    cftime_range\n    date_range_like\n    '
    from xarray.coding.times import _is_standard_calendar
    if tz is not None:
        use_cftime = False
    inclusive = _infer_inclusive(closed, inclusive)
    if _is_standard_calendar(calendar) and use_cftime is not True:
        try:
            return pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, inclusive=inclusive)
        except pd.errors.OutOfBoundsDatetime as err:
            if use_cftime is False:
                raise ValueError('Date range is invalid for pandas DatetimeIndex, try using `use_cftime=True`.') from err
    elif use_cftime is False:
        raise ValueError(f'Invalid calendar {calendar} for pandas DatetimeIndex, try using `use_cftime=True`.')
    return cftime_range(start=start, end=end, periods=periods, freq=freq, normalize=normalize, name=name, inclusive=inclusive, calendar=calendar)

def date_range_like(source, calendar, use_cftime=None):
    if False:
        for i in range(10):
            print('nop')
    "Generate a datetime array with the same frequency, start and end as\n    another one, but in a different calendar.\n\n    Parameters\n    ----------\n    source : DataArray, CFTimeIndex, or pd.DatetimeIndex\n        1D datetime array\n    calendar : str\n        New calendar name.\n    use_cftime : bool, optional\n        If True, the output uses :py:class:`cftime.datetime` objects.\n        If None (default), :py:class:`numpy.datetime64` values are used if possible.\n        If False, :py:class:`numpy.datetime64` values are used or an error is raised.\n\n    Returns\n    -------\n    DataArray\n        1D datetime coordinate with the same start, end and frequency as the\n        source, but in the new calendar. The start date is assumed to exist in\n        the target calendar. If the end date doesn't exist, the code tries 1\n        and 2 calendar days before. There is a special case when the source time\n        series is daily or coarser and the end of the input range is on the\n        last day of the month. Then the output range will also end on the last\n        day of the month in the new calendar.\n    "
    from xarray.coding.frequencies import infer_freq
    from xarray.core.dataarray import DataArray
    if not isinstance(source, (pd.DatetimeIndex, CFTimeIndex)) and (isinstance(source, DataArray) and source.ndim != 1 or not _contains_datetime_like_objects(source.variable)):
        raise ValueError("'source' must be a 1D array of datetime objects for inferring its range.")
    freq = infer_freq(source)
    if freq is None:
        raise ValueError('`date_range_like` was unable to generate a range as the source frequency was not inferable.')
    use_cftime = _should_cftime_be_used(source, calendar, use_cftime)
    source_start = source.values.min()
    source_end = source.values.max()
    if is_np_datetime_like(source.dtype):
        source_calendar = 'standard'
        source_start = nanosecond_precision_timestamp(source_start)
        source_end = nanosecond_precision_timestamp(source_end)
    elif isinstance(source, CFTimeIndex):
        source_calendar = source.calendar
    else:
        source_calendar = source.dt.calendar
    if calendar == source_calendar and is_np_datetime_like(source.dtype) ^ use_cftime:
        return source
    date_type = get_date_type(calendar, use_cftime)
    start = convert_time_or_go_back(source_start, date_type)
    end = convert_time_or_go_back(source_end, date_type)
    if source_end.day == source_end.daysinmonth and isinstance(to_offset(freq), (YearEnd, QuarterEnd, MonthEnd, Day)):
        end = end.replace(day=end.daysinmonth)
    return date_range(start=start.isoformat(), end=end.isoformat(), freq=freq, calendar=calendar)