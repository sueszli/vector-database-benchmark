"""jc - JSON Convert ISO 8601 Datetime string parser

This parser supports standard ISO 8601 strings that include both date and
time. If no timezone or offset information is available in the string, then
UTC timezone is used.

Usage (cli):

    $ echo "2022-07-20T14:52:45Z" | jc --iso-datetime

Usage (module):

    import jc
    result = jc.parse('iso_datetime', iso_8601_string)

Schema:

    {
      "year":               integer,
      "month":              string,
      "month_num":          integer,
      "day":                integer,
      "weekday":            string,
      "weekday_num":        integer,
      "hour":               integer,
      "hour_24":            integer,
      "minute":             integer,
      "second":             integer,
      "microsecond":        integer,
      "period":             string,
      "utc_offset":         string,
      "day_of_year":        integer,
      "week_of_year":       integer,
      "iso":                string,
      "timestamp":          integer  # [0]
    }

    [0] timezone aware UNIX timestamp expressed in UTC

Examples:

    $ echo "2022-07-20T14:52:45Z" | jc --iso-datetime -p
    {
      "year": 2022,
      "month": "Jul",
      "month_num": 7,
      "day": 20,
      "weekday": "Wed",
      "weekday_num": 3,
      "hour": 2,
      "hour_24": 14,
      "minute": 52,
      "second": 45,
      "microsecond": 0,
      "period": "PM",
      "utc_offset": "+0000",
      "day_of_year": 201,
      "week_of_year": 29,
      "iso": "2022-07-20T14:52:45+00:00",
      "timestamp": 1658328765
    }
"""
import datetime
import re
import typing
from decimal import Decimal
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = 'ISO 8601 Datetime string parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    details = 'Using the pyiso8601 library from https://github.com/micktwomey/pyiso8601/releases/tag/1.0.2'
    compatible = ['linux', 'aix', 'freebsd', 'darwin', 'win32', 'cygwin']
    tags = ['standard', 'string']
__version__ = info.version
'\npyiso8601 library from https://github.com/micktwomey/pyiso8601/releases/tag/1.0.2\n'
'\nCopyright (c) 2007 - 2022 Michael Twomey\n\nPermission is hereby granted, free of charge, to any person obtaining a\ncopy of this software and associated documentation files (the\n"Software"), to deal in the Software without restriction, including\nwithout limitation the rights to use, copy, modify, merge, publish,\ndistribute, sublicense, and/or sell copies of the Software, and to\npermit persons to whom the Software is furnished to do so, subject to\nthe following conditions:\n\nThe above copyright notice and this permission notice shall be included\nin all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS\nOR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\nMERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\nIN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY\nCLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,\nTORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE\nSOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n'
'ISO 8601 date time string parsing\nBasic usage:\n>>> import iso8601\n>>> iso8601._parse_date("2007-01-25T12:00:00Z")\ndatetime.datetime(2007, 1, 25, 12, 0, tzinfo=<iso8601.Utc ...>)\n>>>\n'
ISO8601_REGEX = re.compile("\n    (?P<year>[0-9]{4})\n    (\n        (\n            (-(?P<monthdash>[0-9]{1,2}))\n            |\n            (?P<month>[0-9]{2})\n            (?!$)  # Don't allow YYYYMM\n        )\n        (\n            (\n                (-(?P<daydash>[0-9]{1,2}))\n                |\n                (?P<day>[0-9]{2})\n            )\n            (\n                (\n                    (?P<separator>[ T])\n                    (?P<hour>[0-9]{2})\n                    (:{0,1}(?P<minute>[0-9]{2})){0,1}\n                    (\n                        :{0,1}(?P<second>[0-9]{1,2})\n                        ([.,](?P<second_fraction>[0-9]+)){0,1}\n                    ){0,1}\n                    (?P<timezone>\n                        Z\n                        |\n                        (\n                            (?P<tz_sign>[-+])\n                            (?P<tz_hour>[0-9]{2})\n                            :{0,1}\n                            (?P<tz_minute>[0-9]{2}){0,1}\n                        )\n                    ){0,1}\n                ){0,1}\n            )\n        ){0,1}  # YYYY-MM\n    ){0,1}  # YYYY only\n    $\n    ", re.VERBOSE)

class _ParseError(ValueError):
    """Raised when there is a problem parsing a date string"""
UTC = datetime.timezone.utc

def _FixedOffset(offset_hours: float, offset_minutes: float, name: str) -> datetime.timezone:
    if False:
        while True:
            i = 10
    return datetime.timezone(datetime.timedelta(hours=offset_hours, minutes=offset_minutes), name)

def _parse_timezone(matches: typing.Dict[str, str], default_timezone: typing.Optional[datetime.timezone]=UTC) -> typing.Optional[datetime.timezone]:
    if False:
        for i in range(10):
            print('nop')
    'Parses ISO 8601 time zone specs into tzinfo offsets'
    tz = matches.get('timezone', None)
    if tz == 'Z':
        return UTC
    if tz is None:
        return default_timezone
    sign = matches.get('tz_sign', None)
    hours = int(matches.get('tz_hour', 0))
    minutes = int(matches.get('tz_minute', 0))
    description = f'{sign}{hours:02d}:{minutes:02d}'
    if sign == '-':
        hours = -hours
        minutes = -minutes
    return _FixedOffset(hours, minutes, description)

def _parse_date(datestring: str, default_timezone: typing.Optional[datetime.timezone]=UTC) -> datetime.datetime:
    if False:
        print('Hello World!')
    'Parses ISO 8601 dates into datetime objects\n    The timezone is parsed from the date string. However it is quite common to\n    have dates without a timezone (not strictly correct). In this case the\n    default timezone specified in default_timezone is used. This is UTC by\n    default.\n    :param datestring: The date to parse as a string\n    :param default_timezone: A datetime tzinfo instance to use when no timezone\n                             is specified in the datestring. If this is set to\n                             None then a naive datetime object is returned.\n    :returns: A datetime.datetime instance\n    :raises: _ParseError when there is a problem parsing the date or\n             constructing the datetime instance.\n    '
    try:
        m = ISO8601_REGEX.match(datestring)
    except Exception as e:
        raise _ParseError(e)
    if not m:
        raise _ParseError(f'Unable to parse date string {datestring!r}')
    groups: typing.Dict[str, str] = {k: v for (k, v) in m.groupdict().items() if v is not None}
    try:
        return datetime.datetime(year=int(groups.get('year', 0)), month=int(groups.get('month', groups.get('monthdash', 1))), day=int(groups.get('day', groups.get('daydash', 1))), hour=int(groups.get('hour', 0)), minute=int(groups.get('minute', 0)), second=int(groups.get('second', 0)), microsecond=int(Decimal(f"0.{groups.get('second_fraction', 0)}") * Decimal('1000000.0')), tzinfo=_parse_timezone(groups, default_timezone=default_timezone))
    except Exception as e:
        raise _ParseError(e)

def _process(proc_data):
    if False:
        return 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured data to conform to the schema.\n    '
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        return 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output = {}
    if jc.utils.has_data(data):
        dt = _parse_date(data)
        raw_output = {'year': dt.year, 'month': dt.strftime('%b'), 'month_num': dt.month, 'day': dt.day, 'weekday': dt.strftime('%a'), 'weekday_num': dt.isoweekday(), 'hour': int(dt.strftime('%I')), 'hour_24': dt.hour, 'minute': dt.minute, 'second': dt.second, 'microsecond': dt.microsecond, 'period': dt.strftime('%p').upper(), 'utc_offset': dt.strftime('%z') or None, 'day_of_year': int(dt.strftime('%j')), 'week_of_year': int(dt.strftime('%W')), 'iso': dt.isoformat(), 'timestamp': int(dt.timestamp())}
    return raw_output if raw else _process(raw_output)