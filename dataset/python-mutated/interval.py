from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, Any, Union
from dateutil.relativedelta import relativedelta
from pendulum import DateTime
from airflow.exceptions import AirflowTimetableInvalid
from airflow.timetables._cron import CronMixin
from airflow.timetables.base import DagRunInfo, DataInterval, Timetable
from airflow.utils.timezone import convert_to_utc
if TYPE_CHECKING:
    from airflow.timetables.base import TimeRestriction
Delta = Union[datetime.timedelta, relativedelta]

class _DataIntervalTimetable(Timetable):
    """Basis for timetable implementations that schedule data intervals.

    This kind of timetable classes create periodic data intervals from an
    underlying schedule representation (e.g. a cron expression, or a timedelta
    instance), and schedule a DagRun at the end of each interval.
    """

    def _skip_to_latest(self, earliest: DateTime | None) -> DateTime:
        if False:
            return 10
        'Bound the earliest time a run can be scheduled.\n\n        This is called when ``catchup=False``. See docstring of subclasses for\n        exact skipping behaviour of a schedule.\n        '
        raise NotImplementedError()

    def _align_to_next(self, current: DateTime) -> DateTime:
        if False:
            print('Hello World!')
        'Align given time to the next scheduled time.\n\n        For fixed schedules (e.g. every midnight); this finds the next time that\n        aligns to the declared time, if the given time does not align. If the\n        schedule is not fixed (e.g. every hour), the given time is returned.\n        '
        raise NotImplementedError()

    def _align_to_prev(self, current: DateTime) -> DateTime:
        if False:
            for i in range(10):
                print('nop')
        "Align given time to the previous scheduled time.\n\n        For fixed schedules (e.g. every midnight); this finds the prev time that\n        aligns to the declared time, if the given time does not align. If the\n        schedule is not fixed (e.g. every hour), the given time is returned.\n\n        It is not enough to use ``_get_prev(_align_to_next())``, since when a\n        DAG's schedule changes, this alternative would make the first scheduling\n        after the schedule change remain the same.\n        "
        raise NotImplementedError()

    def _get_next(self, current: DateTime) -> DateTime:
        if False:
            while True:
                i = 10
        'Get the first schedule after the current time.'
        raise NotImplementedError()

    def _get_prev(self, current: DateTime) -> DateTime:
        if False:
            while True:
                i = 10
        'Get the last schedule before the current time.'
        raise NotImplementedError()

    def next_dagrun_info(self, *, last_automated_data_interval: DataInterval | None, restriction: TimeRestriction) -> DagRunInfo | None:
        if False:
            return 10
        earliest = restriction.earliest
        if not restriction.catchup:
            earliest = self._skip_to_latest(earliest)
        elif earliest is not None:
            earliest = self._align_to_next(earliest)
        if last_automated_data_interval is None:
            if earliest is None:
                return None
            start = earliest
        else:
            align_last_data_interval_end = self._align_to_prev(last_automated_data_interval.end)
            if earliest is not None:
                start = max(align_last_data_interval_end, earliest)
            else:
                start = align_last_data_interval_end
        if restriction.latest is not None and start > restriction.latest:
            return None
        end = self._get_next(start)
        return DagRunInfo.interval(start=start, end=end)

class CronDataIntervalTimetable(CronMixin, _DataIntervalTimetable):
    """Timetable that schedules data intervals with a cron expression.

    This corresponds to ``schedule=<cron>``, where ``<cron>`` is either
    a five/six-segment representation, or one of ``cron_presets``.

    The implementation extends on croniter to add timezone awareness. This is
    because croniter works only with naive timestamps, and cannot consider DST
    when determining the next/previous time.

    Don't pass ``@once`` in here; use ``OnceTimetable`` instead.
    """

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Timetable:
        if False:
            return 10
        from airflow.serialization.serialized_objects import decode_timezone
        return cls(data['expression'], decode_timezone(data['timezone']))

    def serialize(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        from airflow.serialization.serialized_objects import encode_timezone
        return {'expression': self._expression, 'timezone': encode_timezone(self._timezone)}

    def _skip_to_latest(self, earliest: DateTime | None) -> DateTime:
        if False:
            print('Hello World!')
        'Bound the earliest time a run can be scheduled.\n\n        The logic is that we move start_date up until one period before, so the\n        current time is AFTER the period end, and the job can be created...\n\n        This is slightly different from the delta version at terminal values.\n        If the next schedule should start *right now*, we want the data interval\n        that start now, not the one that ends now.\n        '
        current_time = DateTime.utcnow()
        last_start = self._get_prev(current_time)
        next_start = self._get_next(last_start)
        if next_start == current_time:
            new_start = last_start
        elif next_start > current_time:
            new_start = self._get_prev(last_start)
        else:
            raise AssertionError("next schedule shouldn't be earlier")
        if earliest is None:
            return new_start
        return max(new_start, self._align_to_next(earliest))

    def infer_manual_data_interval(self, *, run_after: DateTime) -> DataInterval:
        if False:
            print('Hello World!')
        end = self._align_to_prev(run_after)
        return DataInterval(start=self._get_prev(end), end=end)

class DeltaDataIntervalTimetable(_DataIntervalTimetable):
    """Timetable that schedules data intervals with a time delta.

    This corresponds to ``schedule=<delta>``, where ``<delta>`` is
    either a ``datetime.timedelta`` or ``dateutil.relativedelta.relativedelta``
    instance.
    """

    def __init__(self, delta: Delta) -> None:
        if False:
            while True:
                i = 10
        self._delta = delta

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Timetable:
        if False:
            return 10
        from airflow.serialization.serialized_objects import decode_relativedelta
        delta = data['delta']
        if isinstance(delta, dict):
            return cls(decode_relativedelta(delta))
        return cls(datetime.timedelta(seconds=delta))

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return if the offsets match.\n\n        This is only for testing purposes and should not be relied on otherwise.\n        '
        if not isinstance(other, DeltaDataIntervalTimetable):
            return NotImplemented
        return self._delta == other._delta

    @property
    def summary(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self._delta)

    def serialize(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        from airflow.serialization.serialized_objects import encode_relativedelta
        delta: Any
        if isinstance(self._delta, datetime.timedelta):
            delta = self._delta.total_seconds()
        else:
            delta = encode_relativedelta(self._delta)
        return {'delta': delta}

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        now = datetime.datetime.now()
        if now + self._delta <= now:
            raise AirflowTimetableInvalid(f'schedule interval must be positive, not {self._delta!r}')

    def _get_next(self, current: DateTime) -> DateTime:
        if False:
            print('Hello World!')
        return convert_to_utc(current + self._delta)

    def _get_prev(self, current: DateTime) -> DateTime:
        if False:
            return 10
        return convert_to_utc(current - self._delta)

    def _align_to_next(self, current: DateTime) -> DateTime:
        if False:
            return 10
        return current

    def _align_to_prev(self, current: DateTime) -> DateTime:
        if False:
            for i in range(10):
                print('nop')
        return current

    @staticmethod
    def _relativedelta_in_seconds(delta: relativedelta) -> int:
        if False:
            return 10
        return delta.years * 365 * 24 * 60 * 60 + delta.months * 30 * 24 * 60 * 60 + delta.days * 24 * 60 * 60 + delta.hours * 60 * 60 + delta.minutes * 60 + delta.seconds

    def _round(self, dt: DateTime) -> DateTime:
        if False:
            return 10
        'Round the given time to the nearest interval.'
        if isinstance(self._delta, datetime.timedelta):
            delta_in_seconds = self._delta.total_seconds()
        else:
            delta_in_seconds = self._relativedelta_in_seconds(self._delta)
        dt_in_seconds = dt.timestamp()
        rounded_dt = dt_in_seconds - dt_in_seconds % delta_in_seconds
        return DateTime.fromtimestamp(rounded_dt, tz=dt.tzinfo)

    def _skip_to_latest(self, earliest: DateTime | None) -> DateTime:
        if False:
            while True:
                i = 10
        'Bound the earliest time a run can be scheduled.\n\n        The logic is that we move start_date up until one period before, so the\n        current time is AFTER the period end, and the job can be created...\n\n        This is slightly different from the cron version at terminal values.\n        '
        round_current_time = self._round(DateTime.utcnow())
        new_start = self._get_prev(round_current_time)
        if earliest is None:
            return new_start
        return max(new_start, earliest)

    def infer_manual_data_interval(self, run_after: DateTime) -> DataInterval:
        if False:
            while True:
                i = 10
        return DataInterval(start=self._get_prev(run_after), end=run_after)