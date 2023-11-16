"""Functions to parse datetime objects."""
import datetime
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_fixed_timezone
date_re = _lazy_re_compile('(?P<year>\\d{4})-(?P<month>\\d{1,2})-(?P<day>\\d{1,2})$')
time_re = _lazy_re_compile('(?P<hour>\\d{1,2}):(?P<minute>\\d{1,2})(?::(?P<second>\\d{1,2})(?:[.,](?P<microsecond>\\d{1,6})\\d{0,6})?)?$')
datetime_re = _lazy_re_compile('(?P<year>\\d{4})-(?P<month>\\d{1,2})-(?P<day>\\d{1,2})[T ](?P<hour>\\d{1,2}):(?P<minute>\\d{1,2})(?::(?P<second>\\d{1,2})(?:[.,](?P<microsecond>\\d{1,6})\\d{0,6})?)?\\s*(?P<tzinfo>Z|[+-]\\d{2}(?::?\\d{2})?)?$')
standard_duration_re = _lazy_re_compile('^(?:(?P<days>-?\\d+) (days?, )?)?(?P<sign>-?)((?:(?P<hours>\\d+):)(?=\\d+:\\d+))?(?:(?P<minutes>\\d+):)?(?P<seconds>\\d+)(?:[.,](?P<microseconds>\\d{1,6})\\d{0,6})?$')
iso8601_duration_re = _lazy_re_compile('^(?P<sign>[-+]?)P(?:(?P<days>\\d+([.,]\\d+)?)D)?(?:T(?:(?P<hours>\\d+([.,]\\d+)?)H)?(?:(?P<minutes>\\d+([.,]\\d+)?)M)?(?:(?P<seconds>\\d+([.,]\\d+)?)S)?)?$')
postgres_interval_re = _lazy_re_compile('^(?:(?P<days>-?\\d+) (days? ?))?(?:(?P<sign>[-+])?(?P<hours>\\d+):(?P<minutes>\\d\\d):(?P<seconds>\\d\\d)(?:\\.(?P<microseconds>\\d{1,6}))?)?$')

def parse_date(value):
    if False:
        i = 10
        return i + 15
    "Parse a string and return a datetime.date.\n\n    Raise ValueError if the input is well formatted but not a valid date.\n    Return None if the input isn't well formatted.\n    "
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        if (match := date_re.match(value)):
            kw = {k: int(v) for (k, v) in match.groupdict().items()}
            return datetime.date(**kw)

def parse_time(value):
    if False:
        print('Hello World!')
    "Parse a string and return a datetime.time.\n\n    This function doesn't support time zone offsets.\n\n    Raise ValueError if the input is well formatted but not a valid time.\n    Return None if the input isn't well formatted, in particular if it\n    contains an offset.\n    "
    try:
        return datetime.time.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        if (match := time_re.match(value)):
            kw = match.groupdict()
            kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
            kw = {k: int(v) for (k, v) in kw.items() if v is not None}
            return datetime.time(**kw)

def parse_datetime(value):
    if False:
        return 10
    "Parse a string and return a datetime.datetime.\n\n    This function supports time zone offsets. When the input contains one,\n    the output uses a timezone with a fixed offset from UTC.\n\n    Raise ValueError if the input is well formatted but not a valid datetime.\n    Return None if the input isn't well formatted.\n    "
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        if (match := datetime_re.match(value)):
            kw = match.groupdict()
            kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
            tzinfo = kw.pop('tzinfo')
            if tzinfo == 'Z':
                tzinfo = datetime.timezone.utc
            elif tzinfo is not None:
                offset_mins = int(tzinfo[-2:]) if len(tzinfo) > 3 else 0
                offset = 60 * int(tzinfo[1:3]) + offset_mins
                if tzinfo[0] == '-':
                    offset = -offset
                tzinfo = get_fixed_timezone(offset)
            kw = {k: int(v) for (k, v) in kw.items() if v is not None}
            return datetime.datetime(**kw, tzinfo=tzinfo)

def parse_duration(value):
    if False:
        return 10
    "Parse a duration string and return a datetime.timedelta.\n\n    The preferred format for durations in Django is '%d %H:%M:%S.%f'.\n\n    Also supports ISO 8601 representation and PostgreSQL's day-time interval\n    format.\n    "
    match = standard_duration_re.match(value) or iso8601_duration_re.match(value) or postgres_interval_re.match(value)
    if match:
        kw = match.groupdict()
        sign = -1 if kw.pop('sign', '+') == '-' else 1
        if kw.get('microseconds'):
            kw['microseconds'] = kw['microseconds'].ljust(6, '0')
        kw = {k: float(v.replace(',', '.')) for (k, v) in kw.items() if v is not None}
        days = datetime.timedelta(kw.pop('days', 0.0) or 0.0)
        if match.re == iso8601_duration_re:
            days *= sign
        return days + sign * datetime.timedelta(**kw)