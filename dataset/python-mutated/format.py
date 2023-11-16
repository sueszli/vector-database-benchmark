"""
The mycroft.util.format module provides various formatting functions for
things like numbers, times, etc.

The focus of these formatting functions is to create human digestible content
either as speech or in display form. It is also enables localization.

The module uses lingua-franca (https://github.com/mycroftai/lingua-franca) to
do most of the actual parsing. However methods may be wrapped specifically for
use in Mycroft Skills.
"""
import datetime
import warnings
from calendar import leapdays
from enum import Enum
from lingua_franca import get_default_loc
from lingua_franca.format import join_list, nice_date, nice_date_time, nice_number, nice_time, nice_year, pronounce_number
from lingua_franca.format import NUMBER_TUPLE, DateTimeFormat, date_time_format, expand_options, _translate_word
from padatious.util import expand_parentheses

class TimeResolution(Enum):
    YEARS = 1
    DAYS = 2
    HOURS = 3
    MINUTES = 4
    SECONDS = 5
    MILLISECONDS = 6

def _duration_handler(time1, lang=None, speech=True, *, time2=None, use_years=True, clock=False, resolution=TimeResolution.SECONDS):
    if False:
        i = 10
        return i + 15
    'Convert duration in seconds to a nice spoken timespan.\n\n    Used as a handler by nice_duration and nice_duration_dt.\n\n    Accepts:\n        datetime.timedelta, or\n        seconds (int/float), or\n        2 x datetime.datetime\n\n    Examples:\n       time1 = 60  ->  "1:00" or "one minute"\n       time1 = 163  ->  "2:43" or "two minutes forty three seconds"\n       time1 = timedelta(seconds=120)  ->  "2:00" or "two minutes"\n\n       time1 = datetime(2019, 3, 12),\n       time2 = datetime(2019, 1, 1)  ->  "seventy days"\n\n    Args:\n        time1: int/float seconds, OR datetime.timedelta, OR datetime.datetime\n        time2 (datetime, optional): subtracted from time1 if time1 is datetime\n        lang (str, optional): a BCP-47 language code, None for default\n        speech (bool, opt): format output for speech (True) or display (False)\n        use_years (bool, opt): rtn years and days if True, total days if False\n        clock (bool, opt): always format output like digital clock (see below)\n        resolution (mycroft.util.format.TimeResolution, optional): lower bound\n\n            mycroft.util.format.TimeResolution values:\n                TimeResolution.YEARS\n                TimeResolution.DAYS\n                TimeResolution.HOURS\n                TimeResolution.MINUTES\n                TimeResolution.SECONDS\n                TimeResolution.MILLISECONDS\n            NOTE: nice_duration will not produce milliseconds\n            unless that resolution is passed.\n\n            NOTE: clock will produce digital clock-like output appropriate to\n            resolution. Has no effect on resolutions DAYS or YEARS. Only\n            applies to displayed output.\n\n    Returns:\n        str: timespan as a string\n    '
    lang = lang or get_default_loc()
    _leapdays = 0
    _input_resolution = resolution
    milliseconds = 0
    type1 = type(time1)
    if time2:
        type2 = type(time2)
        if type1 is not type2:
            raise Exception("nice_duration() can't combine data types: {} and {}".format(type1, type2))
        elif type1 is datetime.datetime:
            duration = time1 - time2
            _leapdays = abs(leapdays(time1.year, time2.year))
            if all([time1.second == 0, time2.second == 0, resolution.value >= TimeResolution.SECONDS.value]):
                resolution = TimeResolution.MINUTES
            if all([time1.minute == 0, time2.minute == 0, resolution.value == TimeResolution.MINUTES.value]):
                resolution = TimeResolution.HOURS
            if all([time1.hour == 0, time2.hour == 0, resolution.value == TimeResolution.HOURS.value]):
                resolution = TimeResolution.DAYS
        else:
            _tmp = warnings.formatwarning
            warnings.formatwarning = lambda msg, *args, **kwargs: '{}\n'.format(msg)
            warning = "WARN: mycroft.util.format.nice_duration_dt() can't subtract " + str(type1) + ". Ignoring 2nd argument '" + str(time2) + "'."
            warnings.warn(warning)
            warnings.formatwarning = _tmp
            duration = time1
    else:
        duration = time1
    if isinstance(duration, float):
        milliseconds = str(duration).split('.')[1]
        if speech:
            milliseconds = milliseconds[:2]
        else:
            milliseconds = milliseconds[:3]
        milliseconds = float('0.' + milliseconds)
    if not isinstance(duration, datetime.timedelta):
        duration = datetime.timedelta(seconds=duration)
    days = duration.days
    if use_years:
        days -= _leapdays if days > 365 else 0
        years = days // 365
    else:
        years = 0
    days = days % 365 if years > 0 else days
    seconds = duration.seconds
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    if speech:
        out = ''
        if years > 0:
            out += pronounce_number(years, lang) + ' '
            out += _translate_word('year' if years == 1 else 'years', lang)
        if days > 0 and resolution.value > TimeResolution.YEARS.value:
            if out:
                out += ' '
            out += pronounce_number(days, lang) + ' '
            out += _translate_word('day' if days == 1 else 'days', lang)
        if hours > 0 and resolution.value > TimeResolution.DAYS.value:
            if out:
                out += ' '
            out += pronounce_number(hours, lang) + ' '
            out += _translate_word('hour' if hours == 1 else 'hours', lang)
        if minutes > 0 and resolution.value > TimeResolution.HOURS.value:
            if out:
                out += ' '
            out += pronounce_number(minutes, lang) + ' '
            out += _translate_word('minute' if minutes == 1 else 'minutes', lang)
        if seconds > 0 and resolution.value >= TimeResolution.SECONDS.value or (milliseconds > 0 and resolution.value == TimeResolution.MILLISECONDS.value):
            if resolution.value == TimeResolution.MILLISECONDS.value:
                seconds += milliseconds
            if out:
                out += ' '
                if len(out.split()) > 3 or seconds < 1:
                    out += _translate_word('and', lang) + ' '
            out += pronounce_number(seconds, lang) + ' '
            out += _translate_word('second' if seconds == 1 else 'seconds', lang)
    else:
        _seconds_str = '0' + str(seconds) if seconds < 10 else str(seconds)
        out = ''
        if years > 0:
            out = str(years) + 'y '
        if days > 0 and resolution.value > TimeResolution.YEARS.value:
            out += str(days) + 'd '
        if hours > 0 and resolution.value > TimeResolution.DAYS.value or (clock and resolution is TimeResolution.HOURS):
            out += str(hours)
        if resolution.value == TimeResolution.MINUTES.value and (not clock):
            out += 'h ' + str(minutes) + 'm' if hours > 0 else str(minutes) + 'm'
        elif minutes > 0 and resolution.value > TimeResolution.HOURS.value or (clock and resolution.value >= TimeResolution.HOURS.value):
            if hours != 0 or (clock and resolution is TimeResolution.HOURS):
                out += ':'
                if minutes < 10:
                    out += '0'
            out += str(minutes) + ':'
            if seconds > 0 and resolution.value > TimeResolution.MINUTES.value or clock:
                out += _seconds_str
            else:
                out += '00'
        elif (seconds > 0 or clock) and resolution.value > TimeResolution.MINUTES.value:
            try:
                if str(hours) == out.split()[-1]:
                    out += ':'
            except IndexError:
                pass
            out += ('00:' if hours > 0 else '0:') + _seconds_str
        if (milliseconds > 0 or clock) and resolution.value == TimeResolution.MILLISECONDS.value:
            _mill = str(milliseconds).split('.')[1]
            while len(_mill) < 3:
                _mill += '0'
            if out == '':
                out = '0:00'
            elif str(hours) == out.split()[-1] and ':' not in out:
                out += ':00:00'
            if ':' in out:
                out += '.' + _mill
        if out and all([resolution.value >= TimeResolution.HOURS.value, ':' not in out, out[-1] != 'm', hours > 0]):
            out += 'h'
        out = out.strip()
    if not out:
        out = 'zero ' if speech else '0'
        if _input_resolution == TimeResolution.YEARS:
            out += 'years' if speech else 'y'
        elif _input_resolution == TimeResolution.DAYS:
            out += 'days' if speech else 'd'
        elif _input_resolution == TimeResolution.HOURS:
            out += 'hours' if speech else 'h'
        elif _input_resolution == TimeResolution.MINUTES:
            if speech:
                out = 'under a minute' if seconds > 0 else 'zero minutes'
            else:
                out = '0m'
        else:
            out = 'zero seconds' if speech else '0:00'
    return out

def nice_duration(duration, lang=None, speech=True, use_years=True, clock=False, resolution=TimeResolution.SECONDS):
    if False:
        return 10
    ' Convert duration in seconds to a nice spoken timespan\n\n    Accepts:\n        time, in seconds, or datetime.timedelta\n\n    Examples:\n       duration = 60  ->  "1:00" or "one minute"\n       duration = 163  ->  "2:43" or "two minutes forty three seconds"\n       duration = timedelta(seconds=120)  ->  "2:00" or "two minutes"\n\n    Args:\n        duration (int/float/datetime.timedelta)\n        lang (str, optional): a BCP-47 language code, None for default\n        speech (bool, opt): format output for speech (True) or display (False)\n        use_years (bool, opt): rtn years and days if True, total days if False\n        clock (bool, opt): always format output like digital clock (see below)\n        resolution (mycroft.util.format.TimeResolution, optional): lower bound\n\n            mycroft.util.format.TimeResolution values:\n                TimeResolution.YEARS\n                TimeResolution.DAYS\n                TimeResolution.HOURS\n                TimeResolution.MINUTES\n                TimeResolution.SECONDS\n                TimeResolution.MILLISECONDS\n\n            NOTE: nice_duration will not produce milliseconds\n            unless that resolution is passed.\n\n            NOTE: clock will produce digital clock-like output appropriate to\n            resolution. Has no effect on resolutions DAYS or YEARS. Only\n            applies to displayed output.\n\n    Returns:\n        str: timespan as a string\n    '
    return _duration_handler(duration, lang=lang, speech=speech, use_years=use_years, resolution=resolution, clock=clock)

def nice_duration_dt(date1, date2, lang=None, speech=True, use_years=True, clock=False, resolution=TimeResolution.SECONDS):
    if False:
        i = 10
        return i + 15
    ' Convert duration between datetimes to a nice spoken timespan\n\n    Accepts:\n        2 x datetime.datetime\n\n    Examples:\n        date1 = datetime(2019, 3, 12),\n        date2 = datetime(2019, 1, 1)  ->  "seventy days"\n\n        date1 = datetime(2019, 12, 25, 20, 30),\n        date2 = datetime(2019, 10, 31, 8, 00),\n        speech = False  ->  "55d 12:30"\n\n    Args:\n        date1, date2 (datetime.datetime)\n        lang (str, optional): a BCP-47 language code, None for default\n        speech (bool, opt): format output for speech (True) or display (False)\n        use_years (bool, opt): rtn years and days if True, total days if False\n        clock (bool, opt): always format output like digital clock (see below)\n        resolution (mycroft.util.format.TimeResolution, optional): lower bound\n\n            mycroft.util.format.TimeResolution values:\n                TimeResolution.YEARS\n                TimeResolution.DAYS\n                TimeResolution.HOURS\n                TimeResolution.MINUTES\n                TimeResolution.SECONDS\n\n            NOTE: nice_duration_dt() cannot do TimeResolution.MILLISECONDS\n            This will silently fall back on TimeResolution.SECONDS\n\n            NOTE: clock will produce digital clock-like output appropriate to\n            resolution. Has no effect on resolutions DAYS or YEARS. Only\n            applies to displayed output.\n\n    Returns:\n        str: timespan as a string\n    '
    try:
        big = max(date1, date2)
        small = min(date1, date2)
    except TypeError:
        big = date1
        small = date2
    return _duration_handler(big, lang=lang, speech=speech, time2=small, use_years=use_years, resolution=resolution, clock=clock)