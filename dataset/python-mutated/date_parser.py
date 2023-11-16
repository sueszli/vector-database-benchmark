import calendar
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from time import struct_time
from typing import Optional
import pandas as pd
import parsedatetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from flask_babel import lazy_gettext as _
from holidays import country_holidays
from pyparsing import CaselessKeyword, Forward, Group, Optional as ppOptional, ParseException, ParserElement, ParseResults, pyparsing_common, quotedString, Suppress
from superset.charts.commands.exceptions import TimeDeltaAmbiguousError, TimeRangeAmbiguousError, TimeRangeParseFailError
from superset.constants import LRU_CACHE_MAX_SIZE, NO_TIME_RANGE
ParserElement.enablePackrat()
logger = logging.getLogger(__name__)

def parse_human_datetime(human_readable: str) -> datetime:
    if False:
        return 10
    'Returns ``datetime.datetime`` from human readable strings'
    x_periods = '^\\s*([0-9]+)\\s+(second|minute|hour|day|week|month|quarter|year)s?\\s*$'
    if re.search(x_periods, human_readable, re.IGNORECASE):
        raise TimeRangeAmbiguousError(human_readable)
    try:
        default = datetime(year=datetime.now().year, month=1, day=1)
        dttm = parse(human_readable, default=default)
    except (ValueError, OverflowError) as ex:
        cal = parsedatetime.Calendar()
        (parsed_dttm, parsed_flags) = cal.parseDT(human_readable)
        if parsed_flags == 0:
            logger.debug(ex)
            raise TimeRangeParseFailError(human_readable) from ex
        if parsed_flags & 2 == 0:
            parsed_dttm = parsed_dttm.replace(hour=0, minute=0, second=0)
        dttm = dttm_from_timetuple(parsed_dttm.utctimetuple())
    return dttm

def normalize_time_delta(human_readable: str) -> dict[str, int]:
    if False:
        return 10
    x_unit = '^\\s*([0-9]+)\\s+(second|minute|hour|day|week|month|quarter|year)s?\\s+(ago|later)*$'
    matched = re.match(x_unit, human_readable, re.IGNORECASE)
    if not matched:
        raise TimeDeltaAmbiguousError(human_readable)
    key = matched[2] + 's'
    value = int(matched[1])
    value = -value if matched[3] == 'ago' else value
    return {key: value}

def dttm_from_timetuple(date_: struct_time) -> datetime:
    if False:
        while True:
            i = 10
    return datetime(date_.tm_year, date_.tm_mon, date_.tm_mday, date_.tm_hour, date_.tm_min, date_.tm_sec)

def get_past_or_future(human_readable: Optional[str], source_time: Optional[datetime]=None) -> datetime:
    if False:
        while True:
            i = 10
    cal = parsedatetime.Calendar()
    source_dttm = dttm_from_timetuple(source_time.timetuple() if source_time else datetime.now().timetuple())
    return dttm_from_timetuple(cal.parse(human_readable or '', source_dttm)[0])

def parse_human_timedelta(human_readable: Optional[str], source_time: Optional[datetime]=None) -> timedelta:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ``datetime.timedelta`` from natural language time deltas\n\n    >>> parse_human_timedelta('1 day') == timedelta(days=1)\n    True\n    "
    source_dttm = dttm_from_timetuple(source_time.timetuple() if source_time else datetime.now().timetuple())
    return get_past_or_future(human_readable, source_time) - source_dttm

def parse_past_timedelta(delta_str: str, source_time: Optional[datetime]=None) -> timedelta:
    if False:
        for i in range(10):
            print('nop')
    "\n    Takes a delta like '1 year' and finds the timedelta for that period in\n    the past, then represents that past timedelta in positive terms.\n\n    parse_human_timedelta('1 year') find the timedelta 1 year in the future.\n    parse_past_timedelta('1 year') returns -datetime.timedelta(-365)\n    or datetime.timedelta(365).\n    "
    return -parse_human_timedelta(delta_str if delta_str.startswith('-') else f'-{delta_str}', source_time)

def get_since_until(time_range: Optional[str]=None, since: Optional[str]=None, until: Optional[str]=None, time_shift: Optional[str]=None, relative_start: Optional[str]=None, relative_end: Optional[str]=None) -> tuple[Optional[datetime], Optional[datetime]]:
    if False:
        print('Hello World!')
    'Return `since` and `until` date time tuple from string representations of\n    time_range, since, until and time_shift.\n\n    This function supports both reading the keys separately (from `since` and\n    `until`), as well as the new `time_range` key. Valid formats are:\n\n        - ISO 8601\n        - X days/years/hours/day/year/weeks\n        - X days/years/hours/day/year/weeks ago\n        - X days/years/hours/day/year/weeks from now\n        - freeform\n\n    Additionally, for `time_range` (these specify both `since` and `until`):\n\n        - Last day\n        - Last week\n        - Last month\n        - Last quarter\n        - Last year\n        - No filter\n        - Last X seconds/minutes/hours/days/weeks/months/years\n        - Next X seconds/minutes/hours/days/weeks/months/years\n\n    '
    separator = ' : '
    _relative_start = relative_start if relative_start else 'today'
    _relative_end = relative_end if relative_end else 'today'
    if time_range == NO_TIME_RANGE or time_range == _(NO_TIME_RANGE):
        return (None, None)
    if time_range and time_range.startswith('Last') and (separator not in time_range):
        time_range = time_range + separator + _relative_end
    if time_range and time_range.startswith('Next') and (separator not in time_range):
        time_range = _relative_start + separator + time_range
    if time_range and time_range.startswith('previous calendar week') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, WEEK), WEEK) : DATETRUNC(DATETIME('today'), WEEK)"
    if time_range and time_range.startswith('previous calendar month') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, MONTH), MONTH) : DATETRUNC(DATETIME('today'), MONTH)"
    if time_range and time_range.startswith('previous calendar year') and (separator not in time_range):
        time_range = "DATETRUNC(DATEADD(DATETIME('today'), -1, YEAR), YEAR) : DATETRUNC(DATETIME('today'), YEAR)"
    if time_range and separator in time_range:
        time_range_lookup = [('^last\\s+(day|week|month|quarter|year)$', lambda unit: f"DATEADD(DATETIME('{_relative_start}'), -1, {unit})"), ('^last\\s+([0-9]+)\\s+(second|minute|hour|day|week|month|year)s?$', lambda delta, unit: f"DATEADD(DATETIME('{_relative_start}'), -{int(delta)}, {unit})"), ('^next\\s+([0-9]+)\\s+(second|minute|hour|day|week|month|year)s?$', lambda delta, unit: f"DATEADD(DATETIME('{_relative_end}'), {int(delta)}, {unit})"), ('^(DATETIME.*|DATEADD.*|DATETRUNC.*|LASTDAY.*|HOLIDAY.*)$', lambda text: text)]
        since_and_until_partition = [_.strip() for _ in time_range.split(separator, 1)]
        since_and_until: list[Optional[str]] = []
        for part in since_and_until_partition:
            if not part:
                since_and_until.append(None)
                continue
            matched = False
            for (pattern, fn) in time_range_lookup:
                result = re.search(pattern, part, re.IGNORECASE)
                if result:
                    matched = True
                    since_and_until.append(fn(*result.groups()))
            if not matched:
                since_and_until.append(f"DATETIME('{part}')")
        (_since, _until) = map(datetime_eval, since_and_until)
    else:
        since = since or ''
        if since:
            since = add_ago_to_since(since)
        _since = parse_human_datetime(since) if since else None
        _until = parse_human_datetime(until) if until else parse_human_datetime(_relative_end)
    if time_shift:
        time_delta = parse_past_timedelta(time_shift)
        _since = _since if _since is None else _since - time_delta
        _until = _until if _until is None else _until - time_delta
    if _since and _until and (_since > _until):
        raise ValueError(_('From date cannot be larger than to date'))
    return (_since, _until)

def add_ago_to_since(since: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Backwards compatibility hack. Without this slices with since: 7 days will\n    be treated as 7 days in the future.\n\n    :param str since:\n    :returns: Since with ago added if necessary\n    :rtype: str\n    '
    since_words = since.split(' ')
    grains = ['days', 'years', 'hours', 'day', 'year', 'weeks']
    if len(since_words) == 2 and since_words[1] in grains:
        since += ' ago'
    return since

class EvalText:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            i = 10
            return i + 15
        self.value = tokens[0]

    def eval(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.value[1:-1]

class EvalDateTimeFunc:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.value = tokens[1]

    def eval(self) -> datetime:
        if False:
            i = 10
            return i + 15
        return parse_human_datetime(self.value.eval())

class EvalDateAddFunc:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            i = 10
            return i + 15
        self.value = tokens[1]

    def eval(self) -> datetime:
        if False:
            print('Hello World!')
        (dttm_expression, delta, unit) = self.value
        dttm = dttm_expression.eval()
        if unit.lower() == 'quarter':
            delta = delta * 3
            unit = 'month'
        return dttm + parse_human_timedelta(f'{delta} {unit}s', dttm)

class EvalDateTruncFunc:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            while True:
                i = 10
        self.value = tokens[1]

    def eval(self) -> datetime:
        if False:
            print('Hello World!')
        (dttm_expression, unit) = self.value
        dttm = dttm_expression.eval()
        if unit == 'year':
            dttm = dttm.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == 'quarter':
            dttm = pd.Period(pd.Timestamp(dttm), freq='Q').to_timestamp().to_pydatetime()
        elif unit == 'month':
            dttm = dttm.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'week':
            dttm -= relativedelta(days=dttm.weekday())
            dttm = dttm.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'day':
            dttm = dttm.replace(hour=0, minute=0, second=0, microsecond=0)
        elif unit == 'hour':
            dttm = dttm.replace(minute=0, second=0, microsecond=0)
        elif unit == 'minute':
            dttm = dttm.replace(second=0, microsecond=0)
        else:
            dttm = dttm.replace(microsecond=0)
        return dttm

class EvalLastDayFunc:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            i = 10
            return i + 15
        self.value = tokens[1]

    def eval(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        (dttm_expression, unit) = self.value
        dttm = dttm_expression.eval()
        if unit == 'year':
            return dttm.replace(month=12, day=31, hour=0, minute=0, second=0, microsecond=0)
        if unit == 'month':
            return dttm.replace(day=calendar.monthrange(dttm.year, dttm.month)[1], hour=0, minute=0, second=0, microsecond=0)
        mon = dttm - relativedelta(days=dttm.weekday())
        mon = mon.replace(hour=0, minute=0, second=0, microsecond=0)
        return mon + relativedelta(days=6)

class EvalHolidayFunc:

    def __init__(self, tokens: ParseResults) -> None:
        if False:
            i = 10
            return i + 15
        self.value = tokens[1]

    def eval(self) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        holiday = self.value[0].eval()
        (dttm, country) = [None, None]
        if len(self.value) >= 2:
            dttm = self.value[1].eval()
        if len(self.value) == 3:
            country = self.value[2]
        holiday_year = dttm.year if dttm else parse_human_datetime('today').year
        country = country.eval() if country else 'US'
        holiday_lookup = country_holidays(country, years=[holiday_year], observed=False)
        searched_result = holiday_lookup.get_named(holiday, lookup='istartswith')
        if len(searched_result) > 0:
            return dttm_from_timetuple(searched_result[0].timetuple())
        raise ValueError(_('Unable to find such a holiday: [%(holiday)s]', holiday=holiday))

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def datetime_parser() -> ParseResults:
    if False:
        while True:
            i = 10
    (DATETIME, DATEADD, DATETRUNC, LASTDAY, HOLIDAY, YEAR, QUARTER, MONTH, WEEK, DAY, HOUR, MINUTE, SECOND) = map(CaselessKeyword, 'datetime dateadd datetrunc lastday holiday year quarter month week day hour minute second'.split())
    (lparen, rparen, comma) = map(Suppress, '(),')
    int_operand = pyparsing_common.signed_integer().setName('int_operand')
    text_operand = quotedString.setName('text_operand').setParseAction(EvalText)
    datetime_func = Forward().setName('datetime')
    dateadd_func = Forward().setName('dateadd')
    datetrunc_func = Forward().setName('datetrunc')
    lastday_func = Forward().setName('lastday')
    holiday_func = Forward().setName('holiday')
    date_expr = datetime_func | dateadd_func | datetrunc_func | lastday_func | holiday_func
    datetime_func <<= (DATETIME + lparen + text_operand + rparen).setParseAction(EvalDateTimeFunc)
    dateadd_func <<= (DATEADD + lparen + Group(date_expr + comma + int_operand + comma + (YEAR | QUARTER | MONTH | WEEK | DAY | HOUR | MINUTE | SECOND) + ppOptional(comma)) + rparen).setParseAction(EvalDateAddFunc)
    datetrunc_func <<= (DATETRUNC + lparen + Group(date_expr + comma + (YEAR | QUARTER | MONTH | WEEK | DAY | HOUR | MINUTE | SECOND) + ppOptional(comma)) + rparen).setParseAction(EvalDateTruncFunc)
    lastday_func <<= (LASTDAY + lparen + Group(date_expr + comma + (YEAR | MONTH | WEEK) + ppOptional(comma)) + rparen).setParseAction(EvalLastDayFunc)
    holiday_func <<= (HOLIDAY + lparen + Group(text_operand + ppOptional(comma) + ppOptional(date_expr) + ppOptional(comma) + ppOptional(text_operand) + ppOptional(comma)) + rparen).setParseAction(EvalHolidayFunc)
    return date_expr

def datetime_eval(datetime_expression: Optional[str]=None) -> Optional[datetime]:
    if False:
        i = 10
        return i + 15
    if datetime_expression:
        try:
            return datetime_parser().parseString(datetime_expression)[0].eval()
        except ParseException as ex:
            raise ValueError(ex) from ex
    return None

class DateRangeMigration:
    x_dateunit_in_since = '"time_range":\\s*"\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*\\s:\\s'
    x_dateunit_in_until = '"time_range":\\s*".*\\s:\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*"'
    x_dateunit = '^\\s*[0-9]+\\s+(day|week|month|quarter|year)s?\\s*$'