"""
Tools for working with dates
"""
from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime
_quarter_to_day = {'1': (3, 31), '2': (6, 30), '3': (9, 30), '4': (12, 31), 'I': (3, 31), 'II': (6, 30), 'III': (9, 30), 'IV': (12, 31)}
_mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_months_with_days = lzip(lrange(1, 13), _mdays)
_month_to_day = dict(zip(map(str, lrange(1, 13)), _months_with_days))
_month_to_day.update(dict(zip(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'], _months_with_days)))
_y_pattern = '^\\d?\\d?\\d?\\d$'
_q_pattern = '\n^               # beginning of string\n\\d?\\d?\\d?\\d     # match any number 1-9999, includes leading zeros\n\n(:?q)           # use q or a : as a separator\n\n([1-4]|(I{1,3}V?)) # match 1-4 or I-IV roman numerals\n\n$               # end of string\n'
_m_pattern = '\n^               # beginning of string\n\\d?\\d?\\d?\\d     # match any number 1-9999, includes leading zeros\n\n(:?m)           # use m or a : as a separator\n\n(([1-9][0-2]?)|(I?XI{0,2}|I?VI{0,3}|I{1,3}))  # match 1-12 or\n                                              # I-XII roman numerals\n\n$               # end of string\n'

def _is_leap(year):
    if False:
        return 10
    year = int(year)
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def date_parser(timestr, parserinfo=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Uses dateutil.parser.parse, but also handles monthly dates of the form\n    1999m4, 1999:m4, 1999:mIV, 1999mIV and the same for quarterly data\n    with q instead of m. It is not case sensitive. The default for annual\n    data is the end of the year, which also differs from dateutil.\n    '
    flags = re.IGNORECASE | re.VERBOSE
    if re.search(_q_pattern, timestr, flags):
        (y, q) = timestr.replace(':', '').lower().split('q')
        (month, day) = _quarter_to_day[q.upper()]
        year = int(y)
    elif re.search(_m_pattern, timestr, flags):
        (y, m) = timestr.replace(':', '').lower().split('m')
        (month, day) = _month_to_day[m.upper()]
        year = int(y)
        if _is_leap(y) and month == 2:
            day += 1
    elif re.search(_y_pattern, timestr, flags):
        (month, day) = (12, 31)
        year = int(timestr)
    else:
        return to_datetime(timestr, **kwargs)
    return datetime.datetime(year, month, day)

def date_range_str(start, end=None, length=None):
    if False:
        while True:
            i = 10
    "\n    Returns a list of abbreviated date strings.\n\n    Parameters\n    ----------\n    start : str\n        The first abbreviated date, for instance, '1965q1' or '1965m1'\n    end : str, optional\n        The last abbreviated date if length is None.\n    length : int, optional\n        The length of the returned array of end is None.\n\n    Returns\n    -------\n    date_range : list\n        List of strings\n    "
    flags = re.IGNORECASE | re.VERBOSE
    start = start.lower()
    if re.search(_m_pattern, start, flags):
        annual_freq = 12
        split = 'm'
    elif re.search(_q_pattern, start, flags):
        annual_freq = 4
        split = 'q'
    elif re.search(_y_pattern, start, flags):
        annual_freq = 1
        start += 'a1'
        if end:
            end += 'a1'
        split = 'a'
    else:
        raise ValueError('Date %s not understood' % start)
    (yr1, offset1) = lmap(int, start.replace(':', '').split(split))
    if end is not None:
        end = end.lower()
        (yr2, offset2) = lmap(int, end.replace(':', '').split(split))
    else:
        if not length:
            raise ValueError('length must be provided if end is None')
        yr2 = yr1 + length // annual_freq
        offset2 = length % annual_freq + (offset1 - 1)
    years = [str(yr) for yr in np.repeat(lrange(yr1 + 1, yr2), annual_freq)]
    years = [str(yr1)] * (annual_freq + 1 - offset1) + years
    years = years + [str(yr2)] * offset2
    if split != 'a':
        offset = np.tile(np.arange(1, annual_freq + 1), yr2 - yr1 - 1).astype('a2')
        offset = np.r_[np.arange(offset1, annual_freq + 1).astype('a2'), offset]
        offset = np.r_[offset, np.arange(1, offset2 + 1).astype('a2')]
        date_arr_range = [''.join([i, split, asstr(j)]) for (i, j) in zip(years, offset)]
    else:
        date_arr_range = years
    return date_arr_range

def dates_from_str(dates):
    if False:
        return 10
    "\n    Turns a sequence of date strings and returns a list of datetime.\n\n    Parameters\n    ----------\n    dates : array_like\n        A sequence of abbreviated dates as string. For instance,\n        '1996m1' or '1996Q1'. The datetime dates are at the end of the\n        period.\n\n    Returns\n    -------\n    date_list : ndarray\n        A list of datetime types.\n    "
    return lmap(date_parser, dates)

def dates_from_range(start, end=None, length=None):
    if False:
        return 10
    "\n    Turns a sequence of date strings and returns a list of datetime.\n\n    Parameters\n    ----------\n    start : str\n        The first abbreviated date, for instance, '1965q1' or '1965m1'\n    end : str, optional\n        The last abbreviated date if length is None.\n    length : int, optional\n        The length of the returned array of end is None.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import pandas as pd\n    >>> nobs = 50\n    >>> dates = pd.date_range('1960m1', length=nobs)\n\n\n    Returns\n    -------\n    date_list : ndarray\n        A list of datetime types.\n    "
    dates = date_range_str(start, end, length)
    return dates_from_str(dates)