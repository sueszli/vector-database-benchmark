import functools
import hashlib
import json
import re
from abc import abstractmethod, abstractproperty
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import AbstractSet, Any, Callable, FrozenSet, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Type, Union, cast
import pendulum
import dagster._check as check
from dagster._annotations import PublicAttr, public
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.instance import DynamicPartitionsStore
from dagster._serdes import whitelist_for_serdes
from dagster._serdes.serdes import FieldSerializer
from dagster._utils import utc_datetime_from_timestamp
from dagster._utils.partitions import DEFAULT_HOURLY_FORMAT_WITHOUT_TIMEZONE
from dagster._utils.schedules import cron_string_iterator, is_valid_cron_schedule, reverse_cron_string_iterator
from ..errors import DagsterInvalidDefinitionError, DagsterInvalidDeserializationVersionError
from .partition import DEFAULT_DATE_FORMAT, PartitionedConfig, PartitionsDefinition, PartitionsSubset, ScheduleType, cron_schedule_from_schedule_type_and_offsets
from .partition_key_range import PartitionKeyRange

class DatetimeFieldSerializer(FieldSerializer):
    """Serializes datetime objects to and from floats."""

    def pack(self, datetime: Optional[datetime], **_kwargs) -> Optional[float]:
        if False:
            while True:
                i = 10
        if datetime:
            check.invariant(datetime.tzinfo is not None)
        return datetime.timestamp() if datetime else None

    def unpack(self, datetime_float: Optional[float], **_kwargs) -> Optional[datetime]:
        if False:
            print('Hello World!')
        return utc_datetime_from_timestamp(datetime_float) if datetime_float else None

class TimeWindow(NamedTuple):
    """An interval that is closed at the start and open at the end.

    Attributes:
        start (datetime): A pendulum datetime that marks the start of the window.
        end (datetime): A pendulum datetime that marks the end of the window.
    """
    start: PublicAttr[datetime]
    end: PublicAttr[datetime]

@whitelist_for_serdes(field_serializers={'start': DatetimeFieldSerializer, 'end': DatetimeFieldSerializer}, is_pickleable=False)
class TimeWindowPartitionsDefinition(PartitionsDefinition, NamedTuple('_TimeWindowPartitionsDefinition', [('start', PublicAttr[datetime]), ('timezone', PublicAttr[str]), ('end', PublicAttr[Optional[datetime]]), ('fmt', PublicAttr[str]), ('end_offset', PublicAttr[int]), ('cron_schedule', PublicAttr[str])])):
    """A set of partitions where each partition corresponds to a time window.

    The provided cron_schedule determines the bounds of the time windows. E.g. a cron_schedule of
    "0 0 \\\\* \\\\* \\\\*" will result in daily partitions that start at midnight and end at midnight of the
    following day.

    The string partition_key associated with each partition corresponds to the start of the
    partition's time window.

    The first partition in the set will start on at the first cron_schedule tick that is equal to
    or after the given start datetime. The last partition in the set will end before the current
    time, unless the end_offset argument is set to a positive number.

    Args:
        cron_schedule (str): Determines the bounds of the time windows.
        start (datetime): The first partition in the set will start on at the first cron_schedule
            tick that is equal to or after this value.
        timezone (Optional[str]): The timezone in which each time should exist.
            Supported strings for timezones are the ones provided by the
            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".
        end (datetime): The last partition (excluding) in the set.
        fmt (str): The date format to use for partition_keys.
        end_offset (int): Extends the partition set by a number of partitions equal to the value
            passed. If end_offset is 0 (the default), the last partition ends before the current
            time. If end_offset is 1, the second-to-last partition ends before the current time,
            and so on.
    """

    def __new__(cls, start: Union[datetime, str], fmt: str, end: Union[datetime, str, None]=None, schedule_type: Optional[ScheduleType]=None, timezone: Optional[str]=None, end_offset: int=0, minute_offset: Optional[int]=None, hour_offset: Optional[int]=None, day_offset: Optional[int]=None, cron_schedule: Optional[str]=None):
        if False:
            print('Hello World!')
        check.opt_str_param(timezone, 'timezone')
        timezone = timezone or 'UTC'
        if isinstance(start, datetime):
            start_dt = pendulum.instance(start, tz=timezone)
            if start.tzinfo:
                start_dt = start_dt.in_tz(tz=timezone)
        else:
            start_dt = pendulum.instance(datetime.strptime(start, fmt), tz=timezone)
        if not end:
            end_dt = None
        elif isinstance(end, datetime):
            end_dt = pendulum.instance(end, tz=timezone)
        else:
            end_dt = pendulum.instance(datetime.strptime(end, fmt), tz=timezone)
        if cron_schedule is not None:
            check.invariant(schedule_type is None and (not minute_offset) and (not hour_offset) and (not day_offset), "If cron_schedule argument is provided, then schedule_type, minute_offset, hour_offset, and day_offset can't also be provided")
        else:
            if schedule_type is None:
                check.failed('One of schedule_type and cron_schedule must be provided')
            cron_schedule = cron_schedule_from_schedule_type_and_offsets(schedule_type=schedule_type, minute_offset=minute_offset or 0, hour_offset=hour_offset or 0, day_offset=day_offset or 0)
        if not is_valid_cron_schedule(cron_schedule):
            raise DagsterInvalidDefinitionError(f"Found invalid cron schedule '{cron_schedule}' for a TimeWindowPartitionsDefinition.")
        return super(TimeWindowPartitionsDefinition, cls).__new__(cls, start_dt, timezone, end_dt, fmt, end_offset, cron_schedule)

    def get_current_timestamp(self, current_time: Optional[datetime]=None) -> float:
        if False:
            print('Hello World!')
        return (pendulum.instance(current_time, tz=self.timezone) if current_time else pendulum.now(self.timezone)).timestamp()

    def get_num_partitions(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> int:
        if False:
            return 10
        current_timestamp = self.get_current_timestamp(current_time=current_time)
        partitions_past_current_time = 0
        num_partitions = 0
        for time_window in self._iterate_time_windows(self.start):
            if self.end and time_window.end.timestamp() > self.end.timestamp():
                break
            if time_window.end.timestamp() <= current_timestamp or partitions_past_current_time < self.end_offset:
                num_partitions += 1
                if time_window.end.timestamp() > current_timestamp:
                    partitions_past_current_time += 1
            else:
                break
        if self.end_offset < 0:
            num_partitions += self.end_offset
        return num_partitions

    def get_partition_keys_between_indexes(self, start_idx: int, end_idx: int, current_time: Optional[datetime]=None) -> List[str]:
        if False:
            print('Hello World!')
        current_timestamp = self.get_current_timestamp(current_time=current_time)
        partitions_past_current_time = 0
        partition_keys = []
        reached_end = False
        for (idx, time_window) in enumerate(self._iterate_time_windows(self.start)):
            if time_window.end.timestamp() >= current_timestamp:
                reached_end = True
            if self.end and time_window.end.timestamp() > self.end.timestamp():
                reached_end = True
            if time_window.end.timestamp() <= current_timestamp or partitions_past_current_time < self.end_offset:
                if idx >= start_idx and idx < end_idx:
                    partition_keys.append(time_window.start.strftime(self.fmt))
                if time_window.end.timestamp() > current_timestamp:
                    partitions_past_current_time += 1
            else:
                break
            if len(partition_keys) >= end_idx - start_idx:
                break
        if reached_end and self.end_offset < 0:
            partition_keys = partition_keys[:self.end_offset]
        return partition_keys

    def get_partition_keys(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Sequence[str]:
        if False:
            print('Hello World!')
        current_timestamp = self.get_current_timestamp(current_time=current_time)
        partitions_past_current_time = 0
        partition_keys: List[str] = []
        for time_window in self._iterate_time_windows(self.start):
            if self.end and time_window.end.timestamp() > self.end.timestamp():
                break
            if time_window.end.timestamp() <= current_timestamp or partitions_past_current_time < self.end_offset:
                partition_keys.append(time_window.start.strftime(self.fmt))
                if time_window.end.timestamp() > current_timestamp:
                    partitions_past_current_time += 1
            else:
                break
        if self.end_offset < 0:
            partition_keys = partition_keys[:self.end_offset]
        return partition_keys

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        schedule_str = self.schedule_type.value.capitalize() if self.schedule_type else self.cron_schedule
        partition_def_str = f'{schedule_str}, starting {self.start.strftime(self.fmt)} {self.timezone}.'
        if self.end_offset != 0:
            partition_def_str += f" End offsetted by {self.end_offset} partition{('' if self.end_offset == 1 else 's')}."
        return partition_def_str

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f"TimeWindowPartitionsDefinition(start={self.start.timestamp()}, timezone='{self.timezone}', fmt='{self.fmt}', end_offset={self.end_offset}, cron_schedule='{self.cron_schedule}')"

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(tuple(self.__repr__()))

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        raise DagsterInvariantViolationError('TimeWindowPartitionsDefinition is not pickleable')

    @functools.lru_cache(maxsize=100)
    def time_window_for_partition_key(self, partition_key: str) -> TimeWindow:
        if False:
            return 10
        partition_key_dt = pendulum.instance(datetime.strptime(partition_key, self.fmt), tz=self.timezone)
        return next(iter(self._iterate_time_windows(partition_key_dt)))

    @functools.lru_cache(maxsize=5)
    def time_windows_for_partition_keys(self, partition_keys: FrozenSet[str], validate: bool=True) -> Sequence[TimeWindow]:
        if False:
            i = 10
            return i + 15
        if len(partition_keys) == 0:
            return []
        sorted_pks = sorted(partition_keys, key=lambda pk: datetime.strptime(pk, self.fmt))
        cur_windows_iterator = iter(self._iterate_time_windows(pendulum.instance(datetime.strptime(sorted_pks[0], self.fmt), tz=self.timezone)))
        partition_key_time_windows: List[TimeWindow] = []
        for partition_key in sorted_pks:
            next_window = next(cur_windows_iterator)
            if next_window.start.strftime(self.fmt) == partition_key:
                partition_key_time_windows.append(next_window)
            else:
                cur_windows_iterator = iter(self._iterate_time_windows(pendulum.instance(datetime.strptime(partition_key, self.fmt), tz=self.timezone)))
                partition_key_time_windows.append(next(cur_windows_iterator))
        if validate:
            start_time_window = self.get_first_partition_window()
            end_time_window = self.get_last_partition_window()
            if start_time_window is None or end_time_window is None:
                check.failed('No partitions in the PartitionsDefinition')
            start_timestamp = start_time_window.start.timestamp()
            end_timestamp = end_time_window.end.timestamp()
            partition_key_time_windows = [tw for tw in partition_key_time_windows if tw.start.timestamp() >= start_timestamp and tw.end.timestamp() <= end_timestamp]
        return partition_key_time_windows

    def start_time_for_partition_key(self, partition_key: str) -> datetime:
        if False:
            i = 10
            return i + 15
        partition_key_dt = pendulum.instance(datetime.strptime(partition_key, self.fmt), tz=self.timezone)
        if self.is_basic_hourly or self.is_basic_daily:
            return partition_key_dt
        return next(iter(self._iterate_time_windows(partition_key_dt))).start

    def get_next_partition_key(self, partition_key: str, current_time: Optional[datetime]=None) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        last_partition_window = self.get_last_partition_window(current_time)
        if last_partition_window is None:
            return None
        partition_key_dt = pendulum.instance(datetime.strptime(partition_key, self.fmt), tz=self.timezone)
        windows_iter = iter(self._iterate_time_windows(partition_key_dt))
        next(windows_iter)
        start_time = next(windows_iter).start
        if start_time >= last_partition_window.end:
            return None
        else:
            return start_time.strftime(self.fmt)

    def get_next_partition_window(self, end_dt: datetime, current_time: Optional[datetime]=None, respect_bounds: bool=True) -> Optional[TimeWindow]:
        if False:
            return 10
        windows_iter = iter(self._iterate_time_windows(end_dt))
        next_window = next(windows_iter)
        if respect_bounds:
            last_partition_window = self.get_last_partition_window(current_time)
            if last_partition_window is None:
                return None
            if next_window.start >= last_partition_window.end:
                return None
        return next_window

    def get_prev_partition_window(self, start_dt: datetime, respect_bounds: bool=True) -> Optional[TimeWindow]:
        if False:
            i = 10
            return i + 15
        windows_iter = iter(self._reverse_iterate_time_windows(start_dt))
        prev_window = next(windows_iter)
        if respect_bounds:
            first_partition_window = self.get_first_partition_window()
            if first_partition_window is None or prev_window.start < first_partition_window.start:
                return None
        return prev_window

    @functools.lru_cache(maxsize=256)
    def _get_first_partition_window(self, *, current_time: datetime) -> Optional[TimeWindow]:
        if False:
            print('Hello World!')
        current_timestamp = current_time.timestamp()
        time_window = next(iter(self._iterate_time_windows(self.start)))
        if self.end_offset == 0:
            return time_window if time_window.end.timestamp() <= current_timestamp else None
        elif self.end_offset > 0:
            iterator = iter(self._iterate_time_windows(current_time))
            curr_window_plus_offset = next(iterator)
            for _ in range(self.end_offset):
                curr_window_plus_offset = next(iterator)
            return time_window if time_window.end.timestamp() <= curr_window_plus_offset.start.timestamp() else None
        else:
            end_window = None
            iterator = iter(self._reverse_iterate_time_windows(current_time))
            for _ in range(abs(self.end_offset)):
                end_window = next(iterator)
            if end_window is None:
                check.failed('end_window should not be None')
            return time_window if time_window.end.timestamp() <= end_window.start.timestamp() else None

    def get_first_partition_window(self, current_time: Optional[datetime]=None) -> Optional[TimeWindow]:
        if False:
            i = 10
            return i + 15
        current_time = cast(datetime, pendulum.instance(current_time, tz=self.timezone) if current_time else pendulum.now(self.timezone))
        return self._get_first_partition_window(current_time=current_time)

    @functools.lru_cache(maxsize=256)
    def _get_last_partition_window(self, *, current_time: datetime) -> Optional[TimeWindow]:
        if False:
            return 10
        if self.get_first_partition_window(current_time) is None:
            return None
        current_time = pendulum.instance(current_time, tz=self.timezone) if current_time else pendulum.now(self.timezone)
        if self.end and self.end < current_time:
            current_time = self.end
        if self.end_offset == 0:
            return next(iter(self._reverse_iterate_time_windows(current_time)))
        else:
            last_partition_key = super().get_last_partition_key(current_time)
            return self.time_window_for_partition_key(last_partition_key) if last_partition_key else None

    def get_last_partition_window(self, current_time: Optional[datetime]=None) -> Optional[TimeWindow]:
        if False:
            for i in range(10):
                print('nop')
        current_time = cast(datetime, pendulum.instance(current_time, tz=self.timezone) if current_time else pendulum.now(self.timezone))
        return self._get_last_partition_window(current_time=current_time)

    def get_first_partition_key(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Optional[str]:
        if False:
            print('Hello World!')
        first_window = self.get_first_partition_window(current_time)
        if first_window is None:
            return None
        return first_window.start.strftime(self.fmt)

    def get_last_partition_key(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        last_window = self.get_last_partition_window(current_time)
        if last_window is None:
            return None
        return last_window.start.strftime(self.fmt)

    def end_time_for_partition_key(self, partition_key: str) -> datetime:
        if False:
            for i in range(10):
                print('nop')
        return self.time_window_for_partition_key(partition_key).end

    @functools.lru_cache(maxsize=5)
    def get_partition_keys_in_time_window(self, time_window: TimeWindow) -> Sequence[str]:
        if False:
            while True:
                i = 10
        result: List[str] = []
        for partition_time_window in self._iterate_time_windows(time_window.start):
            if partition_time_window.start < time_window.end:
                result.append(partition_time_window.start.strftime(self.fmt))
            else:
                break
        return result

    def get_partition_key_range_for_time_window(self, time_window: TimeWindow) -> PartitionKeyRange:
        if False:
            while True:
                i = 10
        start_partition_key = self.get_partition_key_for_timestamp(time_window.start.timestamp())
        end_partition_key = self.get_partition_key_for_timestamp(cast(TimeWindow, self.get_prev_partition_window(time_window.end)).start.timestamp())
        return PartitionKeyRange(start_partition_key, end_partition_key)

    def get_partition_keys_in_range(self, partition_key_range: PartitionKeyRange, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Sequence[str]:
        if False:
            while True:
                i = 10
        start_time = self.start_time_for_partition_key(partition_key_range.start)
        end_time = self.end_time_for_partition_key(partition_key_range.end)
        return self.get_partition_keys_in_time_window(TimeWindow(start_time, end_time))

    @public
    @property
    def schedule_type(self) -> Optional[ScheduleType]:
        if False:
            return 10
        'Optional[ScheduleType]: An enum representing the partition cadence (hourly, daily,\n        weekly, or monthly).\n        '
        if re.fullmatch('\\d+ \\* \\* \\* \\*', self.cron_schedule):
            return ScheduleType.HOURLY
        elif re.fullmatch('\\d+ \\d+ \\* \\* \\*', self.cron_schedule):
            return ScheduleType.DAILY
        elif re.fullmatch('\\d+ \\d+ \\* \\* \\d+', self.cron_schedule):
            return ScheduleType.WEEKLY
        elif re.fullmatch('\\d+ \\d+ \\d+ \\* \\*', self.cron_schedule):
            return ScheduleType.MONTHLY
        else:
            return None

    @public
    @property
    def minute_offset(self) -> int:
        if False:
            i = 10
            return i + 15
        'int: Number of minutes past the hour to "split" partitions. Defaults to 0.\n\n        For example, returns 15 if each partition starts at 15 minutes past the hour.\n        '
        match = re.fullmatch('(\\d+) (\\d+|\\*) (\\d+|\\*) (\\d+|\\*) (\\d+|\\*)', self.cron_schedule)
        if match is None:
            check.failed(f'{self.cron_schedule} has no minute offset')
        return int(match.groups()[0])

    @public
    @property
    def hour_offset(self) -> int:
        if False:
            i = 10
            return i + 15
        'int: Number of hours past 00:00 to "split" partitions. Defaults to 0.\n\n        For example, returns 1 if each partition starts at 01:00.\n        '
        match = re.fullmatch('(\\d+|\\*) (\\d+) (\\d+|\\*) (\\d+|\\*) (\\d+|\\*)', self.cron_schedule)
        if match is None:
            check.failed(f'{self.cron_schedule} has no hour offset')
        return int(match.groups()[1])

    @public
    @property
    def day_offset(self) -> int:
        if False:
            print('Hello World!')
        'int: For a weekly or monthly partitions definition, returns the day to "split" partitions\n        by. Each partition will start on this day, and end before this day in the following\n        week/month. Returns 0 if the day_offset parameter is unset in the\n        WeeklyPartitionsDefinition, MonthlyPartitionsDefinition, or the provided cron schedule.\n\n        For weekly partitions, returns a value between 0 (representing Sunday) and 6 (representing\n        Saturday). Providing a value of 1 means that a partition will exist weekly from Monday to\n        the following Sunday.\n\n        For monthly partitions, returns a value between 0 (the first day of the month) and 31 (the\n        last possible day of the month).\n        '
        schedule_type = self.schedule_type
        if schedule_type == ScheduleType.WEEKLY:
            match = re.fullmatch('(\\d+|\\*) (\\d+|\\*) (\\d+|\\*) (\\d+|\\*) (\\d+)', self.cron_schedule)
            if match is None:
                check.failed(f'{self.cron_schedule} has no day offset')
            return int(match.groups()[4])
        elif schedule_type == ScheduleType.MONTHLY:
            match = re.fullmatch('(\\d+|\\*) (\\d+|\\*) (\\d+) (\\d+|\\*) (\\d+|\\*)', self.cron_schedule)
            if match is None:
                check.failed(f'{self.cron_schedule} has no day offset')
            return int(match.groups()[2])
        else:
            check.failed(f'Unsupported schedule type for day_offset: {schedule_type}')

    @public
    def get_cron_schedule(self, minute_of_hour: Optional[int]=None, hour_of_day: Optional[int]=None, day_of_week: Optional[int]=None, day_of_month: Optional[int]=None) -> str:
        if False:
            while True:
                i = 10
        'The schedule executes at the cadence specified by the partitioning, but may overwrite\n        the minute/hour/day offset of the partitioning.\n\n        This is useful e.g. if you have partitions that span midnight to midnight but you want to\n        schedule a job that runs at 2 am.\n        '
        if minute_of_hour is None and hour_of_day is None and (day_of_week is None) and (day_of_month is None):
            return self.cron_schedule
        schedule_type = self.schedule_type
        if schedule_type is None:
            check.failed(f'{self.cron_schedule} does not support minute_of_hour/hour_of_day/day_of_week/day_of_month arguments')
        minute_of_hour = cast(int, check.opt_int_param(minute_of_hour, 'minute_of_hour', default=self.minute_offset))
        if schedule_type == ScheduleType.HOURLY:
            check.invariant(hour_of_day is None, 'Cannot set hour parameter with hourly partitions.')
        else:
            hour_of_day = cast(int, check.opt_int_param(hour_of_day, 'hour_of_day', default=self.hour_offset))
        if schedule_type == ScheduleType.DAILY:
            check.invariant(day_of_week is None, 'Cannot set day of week parameter with daily partitions.')
            check.invariant(day_of_month is None, 'Cannot set day of month parameter with daily partitions.')
        if schedule_type == ScheduleType.MONTHLY:
            default = self.day_offset or 1
            day_offset = check.opt_int_param(day_of_month, 'day_of_month', default=default)
        elif schedule_type == ScheduleType.WEEKLY:
            default = self.day_offset or 0
            day_offset = check.opt_int_param(day_of_week, 'day_of_week', default=default)
        else:
            day_offset = 0
        return cron_schedule_from_schedule_type_and_offsets(schedule_type, minute_offset=minute_of_hour, hour_offset=hour_of_day or 0, day_offset=day_offset)

    def _iterate_time_windows(self, start: datetime) -> Iterable[TimeWindow]:
        if False:
            while True:
                i = 10
        'Returns an infinite generator of time windows that start after the given start time.'
        start_timestamp = pendulum.instance(start, tz=self.timezone).timestamp()
        iterator = cron_string_iterator(start_timestamp=start_timestamp, cron_string=self.cron_schedule, execution_timezone=self.timezone)
        prev_time = next(iterator)
        while prev_time.timestamp() < start_timestamp:
            prev_time = next(iterator)
        while True:
            next_time = next(iterator)
            yield TimeWindow(prev_time, next_time)
            prev_time = next_time

    def _reverse_iterate_time_windows(self, end: datetime) -> Iterable[TimeWindow]:
        if False:
            while True:
                i = 10
        'Returns an infinite generator of time windows that end before the given end time.'
        end_timestamp = pendulum.instance(end, tz=self.timezone).timestamp()
        iterator = reverse_cron_string_iterator(end_timestamp=end_timestamp, cron_string=self.cron_schedule, execution_timezone=self.timezone)
        prev_time = next(iterator)
        while prev_time.timestamp() > end_timestamp:
            prev_time = next(iterator)
        while True:
            next_time = next(iterator)
            yield TimeWindow(next_time, prev_time)
            prev_time = next_time

    def get_partition_key_for_timestamp(self, timestamp: float, end_closed: bool=False) -> str:
        if False:
            print('Hello World!')
        'Args:\n        timestamp (float): Timestamp from the unix epoch, UTC.\n        end_closed (bool): Whether the interval is closed at the end or at the beginning.\n        '
        iterator = cron_string_iterator(timestamp, self.cron_schedule, self.timezone, start_offset=-1)
        prev = next(iterator)
        prev_next = next(iterator)
        if end_closed or prev_next.timestamp() > timestamp:
            return prev.strftime(self.fmt)
        else:
            return prev_next.strftime(self.fmt)

    def less_than(self, partition_key1: str, partition_key2: str) -> bool:
        if False:
            while True:
                i = 10
        'Returns true if the partition_key1 is earlier than partition_key2.'
        return self.start_time_for_partition_key(partition_key1) < self.start_time_for_partition_key(partition_key2)

    @property
    def partitions_subset_class(self) -> Type['PartitionsSubset']:
        if False:
            return 10
        return PartitionKeysTimeWindowPartitionsSubset

    def empty_subset(self) -> 'PartitionsSubset':
        if False:
            i = 10
            return i + 15
        return self.partitions_subset_class.empty_subset(self)

    def get_serializable_unique_identifier(self, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> str:
        if False:
            while True:
                i = 10
        return hashlib.sha1(self.__repr__().encode('utf-8')).hexdigest()

    def has_partition_key(self, partition_key: str, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> bool:
        if False:
            return 10
        'Returns a boolean representing if the given partition key is valid.'
        try:
            partition_start_time = self.start_time_for_partition_key(partition_key)
        except ValueError:
            return False
        first_partition_window = self.get_first_partition_window(current_time=current_time)
        last_partition_window = self.get_last_partition_window(current_time=current_time)
        return not (first_partition_window is None or last_partition_window is None or partition_start_time < first_partition_window.start or (partition_start_time > last_partition_window.start) or (partition_start_time.strftime(self.fmt) != partition_key))

    def equal_except_for_start_or_end(self, other: 'TimeWindowPartitionsDefinition') -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Returns True iff this is identical to other, except they're allowed to have different\n        start and end datetimes.\n        "
        return self.timezone == other.timezone and self.fmt == other.fmt and (self.cron_schedule == other.cron_schedule) and (self.end_offset == other.end_offset)

    @property
    def is_basic_daily(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.cron_schedule == '0 0 * * *'

    @property
    def is_basic_hourly(self) -> bool:
        if False:
            print('Hello World!')
        return self.cron_schedule == '0 * * * *'

class DailyPartitionsDefinition(TimeWindowPartitionsDefinition):
    """A set of daily partitions.

    The first partition in the set will start at the start_date at midnight. The last partition
    in the set will end before the current time, unless the end_offset argument is set to a
    positive number. If minute_offset and/or hour_offset are used, the start and end times of
    each partition will be hour_offset:minute_offset of each day.

    Args:
        start_date (Union[datetime.datetime, str]): The first date in the set of partitions. Can
            provide in either a datetime or string format.
        end_date (Union[datetime.datetime, str, None]): The last date(excluding) in the set of partitions.
            Default is None. Can provide in either a datetime or string format.
        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults
            to 0.
        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.
        timezone (Optional[str]): The timezone in which each date should exist.
            Supported strings for timezones are the ones provided by the
            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".
        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.
        end_offset (int): Extends the partition set by a number of partitions equal to the value
            passed. If end_offset is 0 (the default), the last partition ends before the current
            time. If end_offset is 1, the second-to-last partition ends before the current time,
            and so on.

    .. code-block:: python

        DailyPartitionsDefinition(start_date="2022-03-12")
        # creates partitions (2022-03-12-00:00, 2022-03-13-00:00), (2022-03-13-00:00, 2022-03-14-00:00), ...

        DailyPartitionsDefinition(start_date="2022-03-12", minute_offset=15, hour_offset=16)
        # creates partitions (2022-03-12-16:15, 2022-03-13-16:15), (2022-03-13-16:15, 2022-03-14-16:15), ...
    """

    def __new__(cls, start_date: Union[datetime, str], end_date: Union[datetime, str, None]=None, minute_offset: int=0, hour_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0):
        if False:
            i = 10
            return i + 15
        _fmt = fmt or DEFAULT_DATE_FORMAT
        return super(DailyPartitionsDefinition, cls).__new__(cls, schedule_type=ScheduleType.DAILY, start=start_date, end=end_date, minute_offset=minute_offset, hour_offset=hour_offset, timezone=timezone, fmt=_fmt, end_offset=end_offset)

def wrap_time_window_run_config_fn(run_config_fn: Optional[Callable[[datetime, datetime], Mapping[str, Any]]], partitions_def: TimeWindowPartitionsDefinition) -> Callable[[str], Mapping[str, Any]]:
    if False:
        i = 10
        return i + 15

    def _run_config_wrapper(key: str) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        if not run_config_fn:
            return {}
        time_window = partitions_def.time_window_for_partition_key(key)
        return run_config_fn(time_window.start, time_window.end)
    return _run_config_wrapper

def wrap_time_window_tags_fn(tags_fn: Optional[Callable[[datetime, datetime], Mapping[str, str]]], partitions_def: TimeWindowPartitionsDefinition) -> Callable[[str], Mapping[str, str]]:
    if False:
        for i in range(10):
            print('nop')

    def _tag_wrapper(key: str) -> Mapping[str, str]:
        if False:
            return 10
        if not tags_fn:
            return {}
        time_window = partitions_def.time_window_for_partition_key(key)
        return tags_fn(time_window.start, time_window.end)
    return _tag_wrapper

def daily_partitioned_config(start_date: Union[datetime, str], minute_offset: int=0, hour_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0, tags_for_partition_fn: Optional[Callable[[datetime, datetime], Mapping[str, str]]]=None) -> Callable[[Callable[[datetime, datetime], Mapping[str, Any]]], PartitionedConfig[DailyPartitionsDefinition]]:
    if False:
        i = 10
        return i + 15
    'Defines run config over a set of daily partitions.\n\n    The decorated function should accept a start datetime and end datetime, which represent the bounds\n    of the date partition the config should delineate.\n\n    The decorated function should return a run config dictionary.\n\n    The resulting object created by this decorator can be provided to the config argument of a Job.\n    The first partition in the set will start at the start_date at midnight. The last partition in\n    the set will end before the current time, unless the end_offset argument is set to a positive\n    number. If minute_offset and/or hour_offset are used, the start and end times of each partition\n    will be hour_offset:minute_offset of each day.\n\n    Args:\n        start_date (Union[datetime.datetime, str]): The first date in the set of partitions. Can\n            provide in either a datetime or string format.\n        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults\n            to 0.\n        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.\n        timezone (Optional[str]): The timezone in which each date should exist.\n            Supported strings for timezones are the ones provided by the\n            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".\n        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.\n        end_offset (int): Extends the partition set by a number of partitions equal to the value\n            passed. If end_offset is 0 (the default), the last partition ends before the current\n            time. If end_offset is 1, the second-to-last partition ends before the current time,\n            and so on.\n        tags_for_partition_fn (Optional[Callable[[str], Mapping[str, str]]]): A function that\n            accepts a partition time window and returns a dictionary of tags to attach to runs for\n            that partition.\n\n    .. code-block:: python\n\n        @daily_partitioned_config(start_date="2022-03-12")\n        # creates partitions (2022-03-12-00:00, 2022-03-13-00:00), (2022-03-13-00:00, 2022-03-14-00:00), ...\n\n        @daily_partitioned_config(start_date="2022-03-12", minute_offset=15, hour_offset=16)\n        # creates partitions (2022-03-12-16:15, 2022-03-13-16:15), (2022-03-13-16:15, 2022-03-14-16:15), ...\n    '

    def inner(fn: Callable[[datetime, datetime], Mapping[str, Any]]) -> PartitionedConfig[DailyPartitionsDefinition]:
        if False:
            return 10
        check.callable_param(fn, 'fn')
        partitions_def = DailyPartitionsDefinition(start_date=start_date, minute_offset=minute_offset, hour_offset=hour_offset, timezone=timezone, fmt=fmt, end_offset=end_offset)
        return PartitionedConfig(run_config_for_partition_key_fn=wrap_time_window_run_config_fn(fn, partitions_def), partitions_def=partitions_def, decorated_fn=fn, tags_for_partition_key_fn=wrap_time_window_tags_fn(tags_for_partition_fn, partitions_def))
    return inner

class HourlyPartitionsDefinition(TimeWindowPartitionsDefinition):
    """A set of hourly partitions.

    The first partition in the set will start on the start_date at midnight. The last partition
    in the set will end before the current time, unless the end_offset argument is set to a
    positive number. If minute_offset is provided, the start and end times of each partition
    will be minute_offset past the hour.

    Args:
        start_date (Union[datetime.datetime, str]): The first date in the set of partitions. Can
            provide in either a datetime or string format.
        end_date (Union[datetime.datetime, str, None]): The last date(excluding) in the set of partitions.
            Default is None. Can provide in either a datetime or string format.
        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults
            to 0.
        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.
        timezone (Optional[str]): The timezone in which each date should exist.
            Supported strings for timezones are the ones provided by the
            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".
        end_offset (int): Extends the partition set by a number of partitions equal to the value
            passed. If end_offset is 0 (the default), the last partition ends before the current
            time. If end_offset is 1, the second-to-last partition ends before the current time,
            and so on.

    .. code-block:: python

        HourlyPartitionsDefinition(start_date=datetime(2022, 03, 12))
        # creates partitions (2022-03-12-00:00, 2022-03-12-01:00), (2022-03-12-01:00, 2022-03-12-02:00), ...

        HourlyPartitionsDefinition(start_date=datetime(2022, 03, 12), minute_offset=15)
        # creates partitions (2022-03-12-00:15, 2022-03-12-01:15), (2022-03-12-01:15, 2022-03-12-02:15), ...
    """

    def __new__(cls, start_date: Union[datetime, str], end_date: Union[datetime, str, None]=None, minute_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0):
        if False:
            print('Hello World!')
        _fmt = fmt or DEFAULT_HOURLY_FORMAT_WITHOUT_TIMEZONE
        return super(HourlyPartitionsDefinition, cls).__new__(cls, schedule_type=ScheduleType.HOURLY, start=start_date, end=end_date, minute_offset=minute_offset, timezone=timezone, fmt=_fmt, end_offset=end_offset)

def hourly_partitioned_config(start_date: Union[datetime, str], minute_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0, tags_for_partition_fn: Optional[Callable[[datetime, datetime], Mapping[str, str]]]=None) -> Callable[[Callable[[datetime, datetime], Mapping[str, Any]]], PartitionedConfig[HourlyPartitionsDefinition]]:
    if False:
        return 10
    'Defines run config over a set of hourly partitions.\n\n    The decorated function should accept a start datetime and end datetime, which represent the date\n    partition the config should delineate.\n\n    The decorated function should return a run config dictionary.\n\n    The resulting object created by this decorator can be provided to the config argument of a Job.\n    The first partition in the set will start at the start_date at midnight. The last partition in\n    the set will end before the current time, unless the end_offset argument is set to a positive\n    number. If minute_offset is provided, the start and end times of each partition will be\n    minute_offset past the hour.\n\n    Args:\n        start_date (Union[datetime.datetime, str]): The first date in the set of partitions. Can\n            provide in either a datetime or string format.\n        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults\n            to 0.\n        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.\n        timezone (Optional[str]): The timezone in which each date should exist.\n            Supported strings for timezones are the ones provided by the\n            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".\n        end_offset (int): Extends the partition set by a number of partitions equal to the value\n            passed. If end_offset is 0 (the default), the last partition ends before the current\n            time. If end_offset is 1, the second-to-last partition ends before the current time,\n            and so on.\n        tags_for_partition_fn (Optional[Callable[[str], Mapping[str, str]]]): A function that\n            accepts a partition time window and returns a dictionary of tags to attach to runs for\n            that partition.\n\n    .. code-block:: python\n\n        @hourly_partitioned_config(start_date=datetime(2022, 03, 12))\n        # creates partitions (2022-03-12-00:00, 2022-03-12-01:00), (2022-03-12-01:00, 2022-03-12-02:00), ...\n\n        @hourly_partitioned_config(start_date=datetime(2022, 03, 12), minute_offset=15)\n        # creates partitions (2022-03-12-00:15, 2022-03-12-01:15), (2022-03-12-01:15, 2022-03-12-02:15), ...\n    '

    def inner(fn: Callable[[datetime, datetime], Mapping[str, Any]]) -> PartitionedConfig[HourlyPartitionsDefinition]:
        if False:
            i = 10
            return i + 15
        check.callable_param(fn, 'fn')
        partitions_def = HourlyPartitionsDefinition(start_date=start_date, minute_offset=minute_offset, timezone=timezone, fmt=fmt, end_offset=end_offset)
        return PartitionedConfig(run_config_for_partition_key_fn=wrap_time_window_run_config_fn(fn, partitions_def), partitions_def=partitions_def, decorated_fn=fn, tags_for_partition_key_fn=wrap_time_window_tags_fn(tags_for_partition_fn, partitions_def))
    return inner

class MonthlyPartitionsDefinition(TimeWindowPartitionsDefinition):
    """A set of monthly partitions.

    The first partition in the set will start at the soonest first of the month after start_date
    at midnight. The last partition in the set will end before the current time, unless the
    end_offset argument is set to a positive number. If day_offset is provided, the start and
    end date of each partition will be day_offset. If minute_offset and/or hour_offset are used,
    the start and end times of each partition will be hour_offset:minute_offset of each day.

    Args:
        start_date (Union[datetime.datetime, str]): The first date in the set of partitions will be
            midnight the sonnest first of the month following start_date. Can provide in either a
            datetime or string format.
        end_date (Union[datetime.datetime, str, None]): The last date(excluding) in the set of partitions.
            Default is None. Can provide in either a datetime or string format.
        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults
            to 0.
        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.
        day_offset (int): Day of the month to "split" the partition. Defaults to 1.
        timezone (Optional[str]): The timezone in which each date should exist.
            Supported strings for timezones are the ones provided by the
            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".
        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.
        end_offset (int): Extends the partition set by a number of partitions equal to the value
            passed. If end_offset is 0 (the default), the last partition ends before the current
            time. If end_offset is 1, the second-to-last partition ends before the current time,
            and so on.

    .. code-block:: python

        MonthlyPartitionsDefinition(start_date="2022-03-12")
        # creates partitions (2022-04-01-00:00, 2022-05-01-00:00), (2022-05-01-00:00, 2022-06-01-00:00), ...

        MonthlyPartitionsDefinition(start_date="2022-03-12", minute_offset=15, hour_offset=3, day_offset=5)
        # creates partitions (2022-04-05-03:15, 2022-05-05-03:15), (2022-05-05-03:15, 2022-06-05-03:15), ...
    """

    def __new__(cls, start_date: Union[datetime, str], end_date: Union[datetime, str, None]=None, minute_offset: int=0, hour_offset: int=0, day_offset: int=1, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0):
        if False:
            while True:
                i = 10
        _fmt = fmt or DEFAULT_DATE_FORMAT
        return super(MonthlyPartitionsDefinition, cls).__new__(cls, schedule_type=ScheduleType.MONTHLY, start=start_date, end=end_date, minute_offset=minute_offset, hour_offset=hour_offset, day_offset=day_offset, timezone=timezone, fmt=_fmt, end_offset=end_offset)

def monthly_partitioned_config(start_date: Union[datetime, str], minute_offset: int=0, hour_offset: int=0, day_offset: int=1, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0, tags_for_partition_fn: Optional[Callable[[datetime, datetime], Mapping[str, str]]]=None) -> Callable[[Callable[[datetime, datetime], Mapping[str, Any]]], PartitionedConfig[MonthlyPartitionsDefinition]]:
    if False:
        while True:
            i = 10
    'Defines run config over a set of monthly partitions.\n\n    The decorated function should accept a start datetime and end datetime, which represent the date\n    partition the config should delineate.\n\n    The decorated function should return a run config dictionary.\n\n    The resulting object created by this decorator can be provided to the config argument of a Job.\n    The first partition in the set will start at midnight on the soonest first of the month after\n    start_date. The last partition in the set will end before the current time, unless the\n    end_offset argument is set to a positive number. If day_offset is provided, the start and end\n    date of each partition will be day_offset. If minute_offset and/or hour_offset are used, the\n    start and end times of each partition will be hour_offset:minute_offset of each day.\n\n    Args:\n        start_date (Union[datetime.datetime, str]): The first date in the set of partitions will be\n            midnight the sonnest first of the month following start_date. Can provide in either a\n            datetime or string format.\n        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults\n            to 0.\n        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.\n        day_offset (int): Day of the month to "split" the partition. Defaults to 1.\n        timezone (Optional[str]): The timezone in which each date should exist.\n            Supported strings for timezones are the ones provided by the\n            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".\n        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.\n        end_offset (int): Extends the partition set by a number of partitions equal to the value\n            passed. If end_offset is 0 (the default), the last partition ends before the current\n            time. If end_offset is 1, the second-to-last partition ends before the current time,\n            and so on.\n        tags_for_partition_fn (Optional[Callable[[str], Mapping[str, str]]]): A function that\n            accepts a partition time window and returns a dictionary of tags to attach to runs for\n            that partition.\n\n    .. code-block:: python\n\n        @monthly_partitioned_config(start_date="2022-03-12")\n        # creates partitions (2022-04-01-00:00, 2022-05-01-00:00), (2022-05-01-00:00, 2022-06-01-00:00), ...\n\n        @monthly_partitioned_config(start_date="2022-03-12", minute_offset=15, hour_offset=3, day_offset=5)\n        # creates partitions (2022-04-05-03:15, 2022-05-05-03:15), (2022-05-05-03:15, 2022-06-05-03:15), ...\n    '

    def inner(fn: Callable[[datetime, datetime], Mapping[str, Any]]) -> PartitionedConfig[MonthlyPartitionsDefinition]:
        if False:
            i = 10
            return i + 15
        check.callable_param(fn, 'fn')
        partitions_def = MonthlyPartitionsDefinition(start_date=start_date, minute_offset=minute_offset, hour_offset=hour_offset, day_offset=day_offset, timezone=timezone, fmt=fmt, end_offset=end_offset)
        return PartitionedConfig(run_config_for_partition_key_fn=wrap_time_window_run_config_fn(fn, partitions_def), partitions_def=partitions_def, decorated_fn=fn, tags_for_partition_key_fn=wrap_time_window_tags_fn(tags_for_partition_fn, partitions_def))
    return inner

class WeeklyPartitionsDefinition(TimeWindowPartitionsDefinition):
    """Defines a set of weekly partitions.

    The first partition in the set will start at the start_date. The last partition in the set will
    end before the current time, unless the end_offset argument is set to a positive number. If
    day_offset is provided, the start and end date of each partition will be day of the week
    corresponding to day_offset (0 indexed with Sunday as the start of the week). If
    minute_offset and/or hour_offset are used, the start and end times of each partition will be
    hour_offset:minute_offset of each day.

    Args:
        start_date (Union[datetime.datetime, str]): The first date in the set of partitions will
            Sunday at midnight following start_date. Can provide in either a datetime or string
            format.
        end_date (Union[datetime.datetime, str, None]): The last date(excluding) in the set of partitions.
            Default is None. Can provide in either a datetime or string format.
        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults
            to 0.
        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.
        day_offset (int): Day of the week to "split" the partition. Defaults to 0 (Sunday).
        timezone (Optional[str]): The timezone in which each date should exist.
            Supported strings for timezones are the ones provided by the
            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".
        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.
        end_offset (int): Extends the partition set by a number of partitions equal to the value
            passed. If end_offset is 0 (the default), the last partition ends before the current
            time. If end_offset is 1, the second-to-last partition ends before the current time,
            and so on.

    .. code-block:: python

        WeeklyPartitionsDefinition(start_date="2022-03-12")
        # creates partitions (2022-03-13-00:00, 2022-03-20-00:00), (2022-03-20-00:00, 2022-03-27-00:00), ...

        WeeklyPartitionsDefinition(start_date="2022-03-12", minute_offset=15, hour_offset=3, day_offset=6)
        # creates partitions (2022-03-12-03:15, 2022-03-19-03:15), (2022-03-19-03:15, 2022-03-26-03:15), ...
    """

    def __new__(cls, start_date: Union[datetime, str], end_date: Union[datetime, str, None]=None, minute_offset: int=0, hour_offset: int=0, day_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0):
        if False:
            return 10
        _fmt = fmt or DEFAULT_DATE_FORMAT
        return super(WeeklyPartitionsDefinition, cls).__new__(cls, schedule_type=ScheduleType.WEEKLY, start=start_date, end=end_date, minute_offset=minute_offset, hour_offset=hour_offset, day_offset=day_offset, timezone=timezone, fmt=_fmt, end_offset=end_offset)

def weekly_partitioned_config(start_date: Union[datetime, str], minute_offset: int=0, hour_offset: int=0, day_offset: int=0, timezone: Optional[str]=None, fmt: Optional[str]=None, end_offset: int=0, tags_for_partition_fn: Optional[Callable[[datetime, datetime], Mapping[str, str]]]=None) -> Callable[[Callable[[datetime, datetime], Mapping[str, Any]]], PartitionedConfig[WeeklyPartitionsDefinition]]:
    if False:
        for i in range(10):
            print('nop')
    'Defines run config over a set of weekly partitions.\n\n    The decorated function should accept a start datetime and end datetime, which represent the date\n    partition the config should delineate.\n\n    The decorated function should return a run config dictionary.\n\n    The resulting object created by this decorator can be provided to the config argument of a Job.\n    The first partition in the set will start at the start_date. The last partition in the set will\n    end before the current time, unless the end_offset argument is set to a positive number. If\n    day_offset is provided, the start and end date of each partition will be day of the week\n    corresponding to day_offset (0 indexed with Sunday as the start of the week). If\n    minute_offset and/or hour_offset are used, the start and end times of each partition will be\n    hour_offset:minute_offset of each day.\n\n    Args:\n        start_date (Union[datetime.datetime, str]): The first date in the set of partitions will\n            Sunday at midnight following start_date. Can provide in either a datetime or string\n            format.\n        minute_offset (int): Number of minutes past the hour to "split" the partition. Defaults\n            to 0.\n        hour_offset (int): Number of hours past 00:00 to "split" the partition. Defaults to 0.\n        day_offset (int): Day of the week to "split" the partition. Defaults to 0 (Sunday).\n        timezone (Optional[str]): The timezone in which each date should exist.\n            Supported strings for timezones are the ones provided by the\n            `IANA time zone database <https://www.iana.org/time-zones>` - e.g. "America/Los_Angeles".\n        fmt (Optional[str]): The date format to use. Defaults to `%Y-%m-%d`.\n        end_offset (int): Extends the partition set by a number of partitions equal to the value\n            passed. If end_offset is 0 (the default), the last partition ends before the current\n            time. If end_offset is 1, the second-to-last partition ends before the current time,\n            and so on.\n        tags_for_partition_fn (Optional[Callable[[str], Mapping[str, str]]]): A function that\n            accepts a partition time window and returns a dictionary of tags to attach to runs for\n            that partition.\n\n    .. code-block:: python\n\n        @weekly_partitioned_config(start_date="2022-03-12")\n        # creates partitions (2022-03-13-00:00, 2022-03-20-00:00), (2022-03-20-00:00, 2022-03-27-00:00), ...\n\n        @weekly_partitioned_config(start_date="2022-03-12", minute_offset=15, hour_offset=3, day_offset=6)\n        # creates partitions (2022-03-12-03:15, 2022-03-19-03:15), (2022-03-19-03:15, 2022-03-26-03:15), ...\n    '

    def inner(fn: Callable[[datetime, datetime], Mapping[str, Any]]) -> PartitionedConfig[WeeklyPartitionsDefinition]:
        if False:
            for i in range(10):
                print('nop')
        check.callable_param(fn, 'fn')
        partitions_def = WeeklyPartitionsDefinition(start_date=start_date, minute_offset=minute_offset, hour_offset=hour_offset, day_offset=day_offset, timezone=timezone, fmt=fmt, end_offset=end_offset)
        return PartitionedConfig(run_config_for_partition_key_fn=wrap_time_window_run_config_fn(fn, partitions_def), partitions_def=partitions_def, decorated_fn=fn, tags_for_partition_key_fn=wrap_time_window_tags_fn(tags_for_partition_fn, partitions_def))
    return inner

class BaseTimeWindowPartitionsSubset(PartitionsSubset):
    """A base class that represents PartitionSubsets for TimeWindowPartitionsDefinitions.
    Contains shared logic for time window partitions subsets, such as building time windows
    from partition keys.
    """
    SERIALIZATION_VERSION = 1

    @abstractproperty
    def included_time_windows(self) -> Sequence[TimeWindow]:
        if False:
            for i in range(10):
                print('nop')
        ...

    @abstractproperty
    def num_partitions(self) -> int:
        if False:
            while True:
                i = 10
        ...

    def _get_partition_time_windows_not_in_subset(self, current_time: Optional[datetime]=None) -> Sequence[TimeWindow]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of partition time windows that are not in the subset.\n        Each time window is a single partition.\n        '
        first_tw = cast(TimeWindowPartitionsDefinition, self.partitions_def).get_first_partition_window(current_time=current_time)
        last_tw = cast(TimeWindowPartitionsDefinition, self.partitions_def).get_last_partition_window(current_time=current_time)
        if not first_tw or not last_tw:
            check.failed('No partitions found')
        if len(self.included_time_windows) == 0:
            return [TimeWindow(first_tw.start, last_tw.end)]
        time_windows = []
        if first_tw.start < self.included_time_windows[0].start:
            time_windows.append(TimeWindow(first_tw.start, self.included_time_windows[0].start))
        for i in range(len(self.included_time_windows) - 1):
            if self.included_time_windows[i].start >= last_tw.end:
                break
            if self.included_time_windows[i].end < last_tw.end:
                if self.included_time_windows[i + 1].start <= last_tw.end:
                    time_windows.append(TimeWindow(self.included_time_windows[i].end, self.included_time_windows[i + 1].start))
                else:
                    time_windows.append(TimeWindow(self.included_time_windows[i].end, last_tw.end))
        if last_tw.end > self.included_time_windows[-1].end:
            time_windows.append(TimeWindow(self.included_time_windows[-1].end, last_tw.end))
        return time_windows

    def get_partition_keys_not_in_subset(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        partition_keys: List[str] = []
        for tw in self._get_partition_time_windows_not_in_subset(current_time):
            partition_keys.extend(cast(TimeWindowPartitionsDefinition, self.partitions_def).get_partition_keys_in_time_window(tw))
        return partition_keys

    @abstractproperty
    def first_start(self) -> datetime:
        if False:
            while True:
                i = 10
        ...

    @abstractproperty
    def is_empty(self) -> bool:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def cheap_ends_before(self, dt: datetime, dt_cron_schedule: str) -> bool:
        if False:
            print('Hello World!')
        ...

    @abstractmethod
    def with_partitions_def(self, partitions_def: TimeWindowPartitionsDefinition) -> 'BaseTimeWindowPartitionsSubset':
        if False:
            i = 10
            return i + 15
        ...

    def get_partition_key_ranges(self, current_time: Optional[datetime]=None, dynamic_partitions_store: Optional[DynamicPartitionsStore]=None) -> Sequence[PartitionKeyRange]:
        if False:
            while True:
                i = 10
        return [cast(TimeWindowPartitionsDefinition, self.partitions_def).get_partition_key_range_for_time_window(window) for window in self.included_time_windows]

    def _add_partitions_to_time_windows(self, initial_windows: Sequence[TimeWindow], partition_keys: Sequence[str], validate: bool=True) -> Tuple[Sequence[TimeWindow], int]:
        if False:
            print('Hello World!')
        'Merges a set of partition keys into an existing set of time windows, returning the\n        minimized set of time windows and the number of partitions added.\n        '
        result_windows = [*initial_windows]
        time_windows = cast(TimeWindowPartitionsDefinition, self.partitions_def).time_windows_for_partition_keys(frozenset(partition_keys), validate=validate)
        num_added_partitions = 0
        for window in sorted(time_windows):
            for i in reversed(range(len(result_windows))):
                included_window = result_windows[i]
                lt_end_of_range = window.start < included_window.end
                gte_start_of_range = window.start >= included_window.start
                if lt_end_of_range and gte_start_of_range:
                    break
                if not lt_end_of_range:
                    merge_with_range = included_window.end == window.start
                    merge_with_later_range = i + 1 < len(result_windows) and window.end == result_windows[i + 1].start
                    if merge_with_range and merge_with_later_range:
                        result_windows[i] = TimeWindow(included_window.start, result_windows[i + 1].end)
                        del result_windows[i + 1]
                    elif merge_with_range:
                        result_windows[i] = TimeWindow(included_window.start, window.end)
                    elif merge_with_later_range:
                        result_windows[i + 1] = TimeWindow(window.start, result_windows[i + 1].end)
                    else:
                        result_windows.insert(i + 1, window)
                    num_added_partitions += 1
                    break
            else:
                if result_windows and window.start == result_windows[0].start:
                    result_windows[0] = TimeWindow(window.start, included_window.end)
                elif result_windows and window.end == result_windows[0].start:
                    result_windows[0] = TimeWindow(window.start, included_window.end)
                else:
                    result_windows.insert(0, window)
                num_added_partitions += 1
        return (result_windows, num_added_partitions)

    def serialize(self) -> str:
        if False:
            i = 10
            return i + 15
        return json.dumps({'version': self.SERIALIZATION_VERSION, 'time_windows': [(window.start.timestamp(), window.end.timestamp()) for window in self.included_time_windows], 'num_partitions': self.num_partitions})

    @classmethod
    def from_serialized(cls, partitions_def: PartitionsDefinition, serialized: str) -> 'PartitionsSubset':
        if False:
            i = 10
            return i + 15
        if not isinstance(partitions_def, TimeWindowPartitionsDefinition):
            check.failed('Partitions definition must be a TimeWindowPartitionsDefinition')
        partitions_def = cast(TimeWindowPartitionsDefinition, partitions_def)
        loaded = json.loads(serialized)

        def tuples_to_time_windows(tuples):
            if False:
                i = 10
                return i + 15
            return [TimeWindow(pendulum.from_timestamp(tup[0], tz=partitions_def.timezone), pendulum.from_timestamp(tup[1], tz=partitions_def.timezone)) for tup in tuples]
        if isinstance(loaded, list):
            time_windows = tuples_to_time_windows(loaded)
            num_partitions = sum((len(partitions_def.get_partition_keys_in_time_window(time_window)) for time_window in time_windows))
        elif isinstance(loaded, dict) and ('version' not in loaded or loaded['version'] == cls.SERIALIZATION_VERSION):
            time_windows = tuples_to_time_windows(loaded['time_windows'])
            num_partitions = loaded['num_partitions']
        else:
            raise DagsterInvalidDeserializationVersionError(f"Attempted to deserialize partition subset with version {loaded.get('version')}, but only version {cls.SERIALIZATION_VERSION} is supported.")
        return TimeWindowPartitionsSubset(partitions_def, num_partitions=num_partitions, included_time_windows=time_windows)

    @classmethod
    def can_deserialize(cls, partitions_def: PartitionsDefinition, serialized: str, serialized_partitions_def_unique_id: Optional[str], serialized_partitions_def_class_name: Optional[str]) -> bool:
        if False:
            return 10
        if serialized_partitions_def_unique_id:
            return partitions_def.get_serializable_unique_identifier() == serialized_partitions_def_unique_id
        if serialized_partitions_def_class_name and serialized_partitions_def_class_name != partitions_def.__class__.__name__:
            return False
        data = json.loads(serialized)
        return isinstance(data, list) or (isinstance(data, dict) and data.get('time_windows') is not None and (data.get('num_partitions') is not None))

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return self.num_partitions

    def __contains__(self, partition_key: str) -> bool:
        if False:
            while True:
                i = 10
        time_window = cast(TimeWindowPartitionsDefinition, self.partitions_def).time_window_for_partition_key(partition_key)
        return any((time_window.start >= included_time_window.start and time_window.start < included_time_window.end for included_time_window in self.included_time_windows))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, BaseTimeWindowPartitionsSubset) and self.partitions_def == other.partitions_def and (self.included_time_windows == other.included_time_windows)

class PartitionKeysTimeWindowPartitionsSubset(BaseTimeWindowPartitionsSubset):
    """A PartitionsSubset for a TimeWindowPartitionsDefinition, which internally represents the
    included partitions using strings.
    """

    def __init__(self, partitions_def: TimeWindowPartitionsDefinition, included_partition_keys: AbstractSet[str]):
        if False:
            while True:
                i = 10
        self._partitions_def = check.inst_param(partitions_def, 'partitions_def', TimeWindowPartitionsDefinition)
        self._included_partition_keys = check.set_param(included_partition_keys, 'included_partition_keys', of_type=str)

    def with_partition_keys(self, partition_keys: Iterable[str]) -> 'BaseTimeWindowPartitionsSubset':
        if False:
            i = 10
            return i + 15
        new_partitions = {*(self._included_partition_keys or []), *partition_keys}
        return PartitionKeysTimeWindowPartitionsSubset(self._partitions_def, included_partition_keys=new_partitions)

    @cached_property
    def included_time_windows(self) -> Sequence[TimeWindow]:
        if False:
            for i in range(10):
                print('nop')
        (result_time_windows, _) = self._add_partitions_to_time_windows(initial_windows=[], partition_keys=list(check.not_none(self._included_partition_keys)), validate=False)
        return result_time_windows

    @property
    def partitions_def(self) -> TimeWindowPartitionsDefinition:
        if False:
            print('Hello World!')
        return self._partitions_def

    @cached_property
    def num_partitions(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._included_partition_keys)

    @public
    def get_partition_keys(self, current_time: Optional[datetime]=None) -> Iterable[str]:
        if False:
            return 10
        return list(self._included_partition_keys) if self._included_partition_keys else []

    @property
    def first_start(self) -> datetime:
        if False:
            print('Hello World!')
        'The start datetime of the earliest partition in the subset.'
        if len(self._included_partition_keys) == 1:
            return self._partitions_def.start_time_for_partition_key(next(iter(self._included_partition_keys)))
        else:
            if len(self.included_time_windows) == 0:
                check.failed(f'Empty subset. self._included_partition_keys: {self._included_partition_keys}')
            return self.included_time_windows[0].start

    @property
    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self._included_partition_keys) == 0

    def cheap_ends_before(self, dt: datetime, dt_cron_schedule: str) -> bool:
        if False:
            print('Hello World!')
        'Performs a cheap calculation that checks whether the latest window in this subset ends\n        before the given dt. If this returns True, then it means the latest window definitely ends\n        before the given dt. If this returns False, it means it may or may not end before the given\n        dt.\n\n        Args:\n            dt_cron_schedule (str): A cron schedule that dt is on one of the ticks of.\n        '
        if len(self._included_partition_keys) == 1:
            if self._partitions_def.cron_schedule == dt_cron_schedule or (self._partitions_def.is_basic_hourly and dt_cron_schedule in ['0 0 * * *', '0 * * * *']):
                return self._partitions_def.start_time_for_partition_key(next(iter(self._included_partition_keys))) < dt
        return False

    def __contains__(self, partition_key: str) -> bool:
        if False:
            i = 10
            return i + 15
        return partition_key in self._included_partition_keys

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, PartitionKeysTimeWindowPartitionsSubset) and self.partitions_def == other.partitions_def and (self._included_partition_keys == other._included_partition_keys) or super(PartitionKeysTimeWindowPartitionsSubset, self).__eq__(other)

    @classmethod
    def empty_subset(cls, partitions_def: PartitionsDefinition) -> 'PartitionsSubset':
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(partitions_def, TimeWindowPartitionsDefinition):
            check.failed('Partitions definition must be a TimeWindowPartitionsDefinition')
        partitions_def = cast(TimeWindowPartitionsDefinition, partitions_def)
        return cls(partitions_def, set())

    def with_partitions_def(self, partitions_def: TimeWindowPartitionsDefinition) -> 'BaseTimeWindowPartitionsSubset':
        if False:
            for i in range(10):
                print('nop')
        check.invariant(partitions_def.cron_schedule == self._partitions_def.cron_schedule, 'num_partitions would become inaccurate if the partitions_defs had different cron schedules')
        return PartitionKeysTimeWindowPartitionsSubset(partitions_def=partitions_def, included_partition_keys=self._included_partition_keys)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'PartitionKeysTimeWindowPartitionsSubset({self.get_partition_key_ranges()})'

class TimeWindowPartitionsSubset(BaseTimeWindowPartitionsSubset):
    """A PartitionsSubset for a TimeWindowPartitionsDefinition, which internally represents the
    included partitions using TimeWindows.
    """

    def __init__(self, partitions_def: TimeWindowPartitionsDefinition, num_partitions: Optional[int], included_time_windows: Sequence[TimeWindow]):
        if False:
            while True:
                i = 10
        self._partitions_def = check.inst_param(partitions_def, 'partitions_def', TimeWindowPartitionsDefinition)
        self._num_partitions = check.opt_int_param(num_partitions, 'num_partitions')
        self._included_time_windows = check.sequence_param(included_time_windows, 'included_time_windows', of_type=TimeWindow)

    @property
    def partitions_def(self) -> TimeWindowPartitionsDefinition:
        if False:
            return 10
        return self._partitions_def

    @property
    def first_start(self) -> datetime:
        if False:
            while True:
                i = 10
        'The start datetime of the earliest partition in the subset.'
        if len(self._included_time_windows) == 0:
            check.failed('Empty subset')
        return self._included_time_windows[0].start

    @property
    def is_empty(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self._included_time_windows) == 0

    def cheap_ends_before(self, dt: datetime, dt_cron_schedule: str) -> bool:
        if False:
            while True:
                i = 10
        'Performs a cheap calculation that checks whether the latest window in this subset ends\n        before the given dt. If this returns True, then it means the latest window definitely ends\n        before the given dt. If this returns False, it means it may or may not end before the given\n        dt.\n\n        Args:\n            dt_cron_schedule (str): A cron schedule that dt is on one of the ticks of.\n        '
        return self._included_time_windows[-1].end <= dt

    @cached_property
    def num_partitions(self) -> int:
        if False:
            while True:
                i = 10
        if self._num_partitions is None:
            return sum((len(self._partitions_def.get_partition_keys_in_time_window(time_window)) for time_window in self.included_time_windows))
        return self._num_partitions

    @property
    def included_time_windows(self) -> Sequence[TimeWindow]:
        if False:
            i = 10
            return i + 15
        return self._included_time_windows

    @classmethod
    def _num_partitions_from_time_windows(cls, partitions_def: TimeWindowPartitionsDefinition, time_windows: Sequence[TimeWindow]) -> int:
        if False:
            return 10
        return sum((len(partitions_def.get_partition_keys_in_time_window(time_window)) for time_window in time_windows))

    @public
    def get_partition_keys(self, current_time: Optional[datetime]=None) -> Iterable[str]:
        if False:
            for i in range(10):
                print('nop')
        return [pk for time_window in self._included_time_windows for pk in self.partitions_def.get_partition_keys_in_time_window(time_window)]

    def with_partition_keys(self, partition_keys: Iterable[str]) -> 'TimeWindowPartitionsSubset':
        if False:
            for i in range(10):
                print('nop')
        (result_windows, added_partitions) = self._add_partitions_to_time_windows(self._included_time_windows, list(partition_keys))
        return TimeWindowPartitionsSubset(self._partitions_def, num_partitions=self.num_partitions + added_partitions, included_time_windows=result_windows)

    @classmethod
    def empty_subset(cls, partitions_def: Optional[PartitionsDefinition]=None) -> 'PartitionsSubset':
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(partitions_def, TimeWindowPartitionsDefinition):
            check.failed('Partitions definition must be a TimeWindowPartitionsDefinition')
        partitions_def = cast(TimeWindowPartitionsDefinition, partitions_def)
        return cls(partitions_def, 0, [])

    def with_partitions_def(self, partitions_def: TimeWindowPartitionsDefinition) -> 'BaseTimeWindowPartitionsSubset':
        if False:
            i = 10
            return i + 15
        check.invariant(partitions_def.cron_schedule == self._partitions_def.cron_schedule, 'num_partitions would become inaccurate if the partitions_defs had different cron schedules')
        return TimeWindowPartitionsSubset(partitions_def=partitions_def, num_partitions=self._num_partitions, included_time_windows=self._included_time_windows)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'TimeWindowPartitionsSubset({self.get_partition_key_ranges()})'

class PartitionRangeStatus(Enum):
    MATERIALIZING = 'MATERIALIZING'
    MATERIALIZED = 'MATERIALIZED'
    FAILED = 'FAILED'
PARTITION_RANGE_STATUS_PRIORITY = [PartitionRangeStatus.MATERIALIZING, PartitionRangeStatus.FAILED, PartitionRangeStatus.MATERIALIZED]

class PartitionTimeWindowStatus:

    def __init__(self, time_window: TimeWindow, status: PartitionRangeStatus):
        if False:
            i = 10
            return i + 15
        self.time_window = time_window
        self.status = status

    def __repr__(self):
        if False:
            return 10
        return f'({self.time_window.start} - {self.time_window.end}): {self.status.value}'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, PartitionTimeWindowStatus) and self.time_window == other.time_window and (self.status == other.status)

def _flatten(high_pri_time_windows: List[PartitionTimeWindowStatus], low_pri_time_windows: List[PartitionTimeWindowStatus]) -> List[PartitionTimeWindowStatus]:
    if False:
        i = 10
        return i + 15
    high_pri_time_windows = sorted(high_pri_time_windows, key=lambda t: t.time_window.start)
    low_pri_time_windows = sorted(low_pri_time_windows, key=lambda t: t.time_window.start)
    high_pri_idx = 0
    low_pri_idx = 0
    filtered_low_pri: List[PartitionTimeWindowStatus] = []
    while True:
        if low_pri_idx >= len(low_pri_time_windows):
            break
        if high_pri_idx >= len(high_pri_time_windows):
            filtered_low_pri.extend(low_pri_time_windows[low_pri_idx:])
            break
        low_pri_tw = low_pri_time_windows[low_pri_idx]
        high_pri_tw = high_pri_time_windows[high_pri_idx]
        if low_pri_tw.time_window.start < high_pri_tw.time_window.start:
            if low_pri_tw.time_window.end <= high_pri_tw.time_window.start:
                filtered_low_pri.append(low_pri_tw)
                low_pri_idx += 1
            else:
                filtered_low_pri.append(PartitionTimeWindowStatus(TimeWindow(low_pri_tw.time_window.start, high_pri_tw.time_window.start), low_pri_tw.status))
                if low_pri_tw.time_window.end > high_pri_tw.time_window.end:
                    low_pri_time_windows[low_pri_idx] = PartitionTimeWindowStatus(TimeWindow(high_pri_tw.time_window.end, low_pri_tw.time_window.end), low_pri_tw.status)
                    high_pri_idx += 1
                else:
                    low_pri_idx += 1
        elif low_pri_tw.time_window.start >= high_pri_tw.time_window.end:
            high_pri_idx += 1
        elif low_pri_tw.time_window.end <= high_pri_tw.time_window.end:
            low_pri_idx += 1
        else:
            low_pri_time_windows[low_pri_idx] = PartitionTimeWindowStatus(TimeWindow(high_pri_tw.time_window.end, low_pri_tw.time_window.end), low_pri_tw.status)
            high_pri_idx += 1
    flattened_time_windows = high_pri_time_windows
    flattened_time_windows.extend(filtered_low_pri)
    flattened_time_windows.sort(key=lambda t: t.time_window.start)
    return flattened_time_windows

def fetch_flattened_time_window_ranges(subsets: Mapping[PartitionRangeStatus, BaseTimeWindowPartitionsSubset]) -> Sequence[PartitionTimeWindowStatus]:
    if False:
        print('Hello World!')
    'Given potentially overlapping subsets, return a flattened list of timewindows where the highest priority status wins\n    on overlaps.\n    '
    prioritized_subsets = sorted([(status, subset) for (status, subset) in subsets.items()], key=lambda t: PARTITION_RANGE_STATUS_PRIORITY.index(t[0]))
    flattened_time_window_statuses = []
    for (status, subset) in prioritized_subsets:
        subset_time_window_statuses = [PartitionTimeWindowStatus(tw, status) for tw in subset.included_time_windows]
        flattened_time_window_statuses = _flatten(flattened_time_window_statuses, subset_time_window_statuses)
    return flattened_time_window_statuses

def has_one_dimension_time_window_partitioning(partitions_def: Optional[PartitionsDefinition]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    from .multi_dimensional_partitions import MultiPartitionsDefinition
    if isinstance(partitions_def, TimeWindowPartitionsDefinition):
        return True
    elif isinstance(partitions_def, MultiPartitionsDefinition):
        time_window_dims = [dim for dim in partitions_def.partitions_defs if isinstance(dim.partitions_def, TimeWindowPartitionsDefinition)]
        if len(time_window_dims) == 1:
            return True
    return False

def get_time_partitions_def(partitions_def: Optional[PartitionsDefinition]) -> Optional[TimeWindowPartitionsDefinition]:
    if False:
        return 10
    'For a given PartitionsDefinition, return the associated TimeWindowPartitionsDefinition if it\n    exists.\n    '
    from .multi_dimensional_partitions import MultiPartitionsDefinition
    if partitions_def is None:
        return None
    elif isinstance(partitions_def, TimeWindowPartitionsDefinition):
        return partitions_def
    elif isinstance(partitions_def, MultiPartitionsDefinition) and has_one_dimension_time_window_partitioning(partitions_def):
        return cast(TimeWindowPartitionsDefinition, partitions_def.time_window_dimension.partitions_def)
    else:
        return None

def get_time_partition_key(partitions_def: Optional[PartitionsDefinition], partition_key: Optional[str]) -> str:
    if False:
        i = 10
        return i + 15
    from .multi_dimensional_partitions import MultiPartitionsDefinition
    if partitions_def is None or partition_key is None:
        check.failed('Cannot get time partitions key from when partitions def is None or partition key is None')
    elif isinstance(partitions_def, TimeWindowPartitionsDefinition):
        return partition_key
    elif isinstance(partitions_def, MultiPartitionsDefinition):
        return partitions_def.get_partition_key_from_str(partition_key).keys_by_dimension[partitions_def.time_window_dimension.name]
    else:
        check.failed(f'Cannot get time partition from non-time partitions def {partitions_def}')