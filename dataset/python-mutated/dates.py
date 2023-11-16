from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from typing import Collection
from croniter import croniter
from dateutil.relativedelta import relativedelta
from airflow.exceptions import RemovedInAirflow3Warning
from airflow.typing_compat import Literal
from airflow.utils import timezone
cron_presets: dict[str, str] = {'@hourly': '0 * * * *', '@daily': '0 0 * * *', '@weekly': '0 0 * * 0', '@monthly': '0 0 1 * *', '@quarterly': '0 0 1 */3 *', '@yearly': '0 0 1 1 *'}

def date_range(start_date: datetime, end_date: datetime | None=None, num: int | None=None, delta: str | timedelta | relativedelta | None=None) -> list[datetime]:
    if False:
        return 10
    'Get a list of dates in the specified range, separated by delta.\n\n    .. code-block:: pycon\n        >>> from airflow.utils.dates import date_range\n        >>> from datetime import datetime, timedelta\n        >>> date_range(datetime(2016, 1, 1), datetime(2016, 1, 3), delta=timedelta(1))\n        [datetime.datetime(2016, 1, 1, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 1, 2, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 1, 3, 0, 0, tzinfo=Timezone(\'UTC\'))]\n        >>> date_range(datetime(2016, 1, 1), datetime(2016, 1, 3), delta="0 0 * * *")\n        [datetime.datetime(2016, 1, 1, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 1, 2, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 1, 3, 0, 0, tzinfo=Timezone(\'UTC\'))]\n        >>> date_range(datetime(2016, 1, 1), datetime(2016, 3, 3), delta="0 0 0 * *")\n        [datetime.datetime(2016, 1, 1, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 2, 1, 0, 0, tzinfo=Timezone(\'UTC\')),\n        datetime.datetime(2016, 3, 1, 0, 0, tzinfo=Timezone(\'UTC\'))]\n\n    :param start_date: anchor date to start the series from\n    :param end_date: right boundary for the date range\n    :param num: alternatively to end_date, you can specify the number of\n        number of entries you want in the range. This number can be negative,\n        output will always be sorted regardless\n    :param delta: step length. It can be datetime.timedelta or cron expression as string\n    '
    warnings.warn('`airflow.utils.dates.date_range()` is deprecated. Please use `airflow.timetables`.', category=RemovedInAirflow3Warning, stacklevel=2)
    if not delta:
        return []
    if end_date:
        if start_date > end_date:
            raise Exception('Wait. start_date needs to be before end_date')
        if num:
            raise Exception('Wait. Either specify end_date OR num')
    if not end_date and (not num):
        end_date = timezone.utcnow()
    delta_iscron = False
    time_zone = start_date.tzinfo
    abs_delta: timedelta | relativedelta
    if isinstance(delta, str):
        delta_iscron = True
        if timezone.is_localized(start_date):
            start_date = timezone.make_naive(start_date, time_zone)
        cron = croniter(cron_presets.get(delta, delta), start_date)
    elif isinstance(delta, timedelta):
        abs_delta = abs(delta)
    elif isinstance(delta, relativedelta):
        abs_delta = abs(delta)
    else:
        raise Exception('Wait. delta must be either datetime.timedelta or cron expression as str')
    dates = []
    if end_date:
        if timezone.is_naive(start_date) and (not timezone.is_naive(end_date)):
            end_date = timezone.make_naive(end_date, time_zone)
        while start_date <= end_date:
            if timezone.is_naive(start_date):
                dates.append(timezone.make_aware(start_date, time_zone))
            else:
                dates.append(start_date)
            if delta_iscron:
                start_date = cron.get_next(datetime)
            else:
                start_date += abs_delta
    else:
        num_entries: int = num
        for _ in range(abs(num_entries)):
            if timezone.is_naive(start_date):
                dates.append(timezone.make_aware(start_date, time_zone))
            else:
                dates.append(start_date)
            if delta_iscron and num_entries > 0:
                start_date = cron.get_next(datetime)
            elif delta_iscron:
                start_date = cron.get_prev(datetime)
            elif num_entries > 0:
                start_date += abs_delta
            else:
                start_date -= abs_delta
    return sorted(dates)

def round_time(dt: datetime, delta: str | timedelta | relativedelta, start_date: datetime=timezone.make_aware(datetime.min)):
    if False:
        for i in range(10):
            print('nop')
    'Return ``start_date + i * delta`` for given ``i`` where the result is closest to ``dt``.\n\n    .. code-block:: pycon\n\n        >>> round_time(datetime(2015, 1, 1, 6), timedelta(days=1))\n        datetime.datetime(2015, 1, 1, 0, 0)\n        >>> round_time(datetime(2015, 1, 2), relativedelta(months=1))\n        datetime.datetime(2015, 1, 1, 0, 0)\n        >>> round_time(datetime(2015, 9, 16, 0, 0), timedelta(1), datetime(2015, 9, 14, 0, 0))\n        datetime.datetime(2015, 9, 16, 0, 0)\n        >>> round_time(datetime(2015, 9, 15, 0, 0), timedelta(1), datetime(2015, 9, 14, 0, 0))\n        datetime.datetime(2015, 9, 15, 0, 0)\n        >>> round_time(datetime(2015, 9, 14, 0, 0), timedelta(1), datetime(2015, 9, 14, 0, 0))\n        datetime.datetime(2015, 9, 14, 0, 0)\n        >>> round_time(datetime(2015, 9, 13, 0, 0), timedelta(1), datetime(2015, 9, 14, 0, 0))\n        datetime.datetime(2015, 9, 14, 0, 0)\n    '
    if isinstance(delta, str):
        time_zone = start_date.tzinfo
        start_date = timezone.make_naive(start_date, time_zone)
        cron = croniter(delta, start_date)
        prev = cron.get_prev(datetime)
        if prev == start_date:
            return timezone.make_aware(start_date, time_zone)
        else:
            return timezone.make_aware(prev, time_zone)
    dt -= timedelta(microseconds=dt.microsecond)
    upper = 1
    while start_date + upper * delta < dt:
        upper *= 2
    lower = upper // 2
    while True:
        if start_date + (lower + 1) * delta >= dt:
            if start_date + (lower + 1) * delta - dt <= dt - (start_date + lower * delta):
                return start_date + (lower + 1) * delta
            else:
                return start_date + lower * delta
        candidate = lower + (upper - lower) // 2
        if start_date + candidate * delta >= dt:
            upper = candidate
        else:
            lower = candidate
TimeUnit = Literal['days', 'hours', 'minutes', 'seconds']

def infer_time_unit(time_seconds_arr: Collection[float]) -> TimeUnit:
    if False:
        print('Hello World!')
    "Determine the most appropriate time unit for given durations (in seconds).\n\n    e.g. 5400 seconds => 'minutes', 36000 seconds => 'hours'\n    "
    if not time_seconds_arr:
        return 'hours'
    max_time_seconds = max(time_seconds_arr)
    if max_time_seconds <= 60 * 2:
        return 'seconds'
    elif max_time_seconds <= 60 * 60 * 2:
        return 'minutes'
    elif max_time_seconds <= 24 * 60 * 60 * 2:
        return 'hours'
    else:
        return 'days'

def scale_time_units(time_seconds_arr: Collection[float], unit: TimeUnit) -> Collection[float]:
    if False:
        for i in range(10):
            print('nop')
    'Convert an array of time durations in seconds to the specified time unit.'
    if unit == 'minutes':
        factor = 60
    elif unit == 'hours':
        factor = 60 * 60
    elif unit == 'days':
        factor = 24 * 60 * 60
    else:
        factor = 1
    return [x / factor for x in time_seconds_arr]

def days_ago(n, hour=0, minute=0, second=0, microsecond=0):
    if False:
        return 10
    'Get a datetime object representing *n* days ago.\n\n    By default the time is set to midnight.\n    '
    warnings.warn("Function `days_ago` is deprecated and will be removed in Airflow 3.0. You can achieve equivalent behavior with `pendulum.today('UTC').add(days=-N, ...)`", RemovedInAirflow3Warning, stacklevel=2)
    today = timezone.utcnow().replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
    return today - timedelta(days=n)

def parse_execution_date(execution_date_str):
    if False:
        while True:
            i = 10
    'Parse execution date string to datetime object.'
    return timezone.parse(execution_date_str)