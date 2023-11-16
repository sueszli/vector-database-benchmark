from datetime import datetime, timedelta
from functools import cached_property
from typing import Dict, Literal, Optional, Tuple
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta
from django.utils import timezone
from posthog.models.filters.base_filter import BaseFilter
from posthog.models.filters import AnyFilter
from posthog.models.filters.mixins.interval import IntervalMixin
from posthog.models.team import Team
from posthog.queries.util import TIME_IN_SECONDS, get_earliest_timestamp, get_start_of_interval_sql
from posthog.utils import DEFAULT_DATE_FROM_DAYS, relative_date_parse, relative_date_parse_with_delta_mapping

class QueryDateRange:
    """Translation of the raw `date_from` and `date_to` filter values to datetimes.

    A raw `date_from` and `date_to` value can either be:
    - unset, in which case `date_from` takes the timestamp of the earliest event in the project and `date_to` equals now
    - a string, which can be a datetime in any format supported by dateutil.parser.isoparse()
    - a datetime already (only for filters constructed internally)
    """
    _filter: AnyFilter
    _team: Team
    _table: str
    _should_round: Optional[bool]

    def __init__(self, filter: AnyFilter, team: Team, should_round: Optional[bool]=None, table='') -> None:
        if False:
            for i in range(10):
                print('nop')
        filter.team = team
        self._filter = filter
        self._team = team
        self._table = f'{table}.' if table else ''
        self._should_round = should_round

    @cached_property
    def date_to_param(self) -> datetime:
        if False:
            return 10
        date_to = self._now
        delta_mapping = None
        position: str | None = None
        if isinstance(self._filter._date_to, str):
            (date_to, delta_mapping, position) = relative_date_parse_with_delta_mapping(self._filter._date_to, self._team.timezone_info)
        elif isinstance(self._filter._date_to, datetime):
            date_to = self._localize_to_team(self._filter._date_to)
        is_relative = not self._filter._date_to or delta_mapping is not None
        if not self._filter.use_explicit_dates:
            if not self.is_hourly(self._filter._date_to):
                date_to = date_to.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif is_relative and (not position):
                date_to = date_to.replace(minute=59, second=59, microsecond=999999)
        return date_to

    def get_earliest_timestamp(self):
        if False:
            i = 10
            return i + 15
        return get_earliest_timestamp(self._team.pk)

    @cached_property
    def date_from_param(self) -> datetime:
        if False:
            i = 10
            return i + 15
        date_from: datetime
        if self._filter._date_from == 'all':
            date_from = self.get_earliest_timestamp()
        elif isinstance(self._filter._date_from, str):
            date_from = relative_date_parse(self._filter._date_from, self._team.timezone_info)
        elif isinstance(self._filter._date_from, datetime):
            date_from = self._localize_to_team(self._filter._date_from)
        else:
            date_from = self._now.replace(hour=0, minute=0, second=0, microsecond=0) - relativedelta(days=DEFAULT_DATE_FROM_DAYS)
        if not self.is_hourly(self._filter._date_from) and (not self._filter.use_explicit_dates):
            date_from = date_from.replace(hour=0, minute=0, second=0, microsecond=0)
        return date_from

    @cached_property
    def _now(self):
        if False:
            return 10
        return self._localize_to_team(timezone.now())

    def _localize_to_team(self, target: datetime):
        if False:
            while True:
                i = 10
        return target.astimezone(ZoneInfo(self._team.timezone))

    @cached_property
    def date_to_clause(self):
        if False:
            while True:
                i = 10
        return self._get_timezone_aware_date_condition('date_to')

    @cached_property
    def date_from_clause(self):
        if False:
            print('Hello World!')
        return self._get_timezone_aware_date_condition('date_from')

    @cached_property
    def date_to(self) -> Tuple[str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        date_to_query = self.date_to_clause
        date_to = self.date_to_param
        date_to_param = {'date_to': date_to.strftime('%Y-%m-%d %H:%M:%S'), 'timezone': self._team.timezone}
        return (date_to_query, date_to_param)

    @cached_property
    def date_from(self) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        date_from_query = self.date_from_clause
        date_from = self.date_from_param
        date_from_param = {'date_from': date_from.strftime('%Y-%m-%d %H:%M:%S'), 'timezone': self._team.timezone}
        return (date_from_query, date_from_param)

    def _get_timezone_aware_date_condition(self, date_param: Literal['date_from', 'date_to']) -> str:
        if False:
            for i in range(10):
                print('nop')
        operator = '>=' if date_param == 'date_from' else '<='
        event_timestamp_expr = self._normalize_datetime(column=f'{self._table}timestamp')
        date_expr = self._normalize_datetime(param=date_param)
        if operator == '>=' and self.should_round:
            if not (isinstance(self._filter, BaseFilter) and isinstance(self._filter, IntervalMixin)):
                raise ValueError("Cannot round with a filter that's not based on BaseFilter with IntervalMixin")
            date_expr = get_start_of_interval_sql(self._filter.interval, team=self._team, source=date_expr, ensure_datetime=True)
        return f'AND {event_timestamp_expr} {operator} {date_expr}'

    @staticmethod
    def _normalize_datetime(*, column: Optional[str]=None, param: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        'Return expression with datetime normalized to project timezone.\n\n        If normalizing a column (such as `events.timestamp`) provide the column expression as `column`\n        (e.g. `"events.timestamp"`). Stored data is already of type `DateTime(\'UTC\')` already, so we just\n        need to convert that to the project TZ.\n        If normalizing a parameter (such as `%(date_from)s`) provide the parameter name as `param` (e.g. `"date_from"`).\n        Such parameters are strings, so they need to be parsed. They\'re assumed to already be in the project TZ.\n        '
        if column and param:
            raise ValueError('Must provide either column or param, not both')
        if column:
            return f'toTimeZone({column}, %(timezone)s)'
        elif param:
            return f'toDateTime(%({param})s, %(timezone)s)'
        else:
            raise ValueError('Must provide either column or param')

    @cached_property
    def delta(self) -> timedelta:
        if False:
            while True:
                i = 10
        return self.date_to_param - self.date_from_param

    @cached_property
    def num_intervals(self) -> int:
        if False:
            print('Hello World!')
        if not hasattr(self._filter, 'interval'):
            return 1
        if self._filter.interval == 'month':
            rel_delta = relativedelta(self.date_to_param, self.date_from_param)
            return rel_delta.years * 12 + rel_delta.months + 1
        return int(self.delta.total_seconds() / TIME_IN_SECONDS[self._filter.interval]) + 1

    @cached_property
    def should_round(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._should_round is not None:
            return self._should_round
        if not hasattr(self._filter, 'interval') or self._filter.use_explicit_dates:
            return False
        round_interval = False
        if self._filter.interval in ['week', 'month']:
            round_interval = True
        else:
            round_interval = self.delta.total_seconds() >= TIME_IN_SECONDS[self._filter.interval] * 2
        return round_interval

    def is_hourly(self, target):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self._filter, 'interval'):
            return False
        return self._filter.interval == 'hour' or (target and isinstance(target, str) and ('h' in target))