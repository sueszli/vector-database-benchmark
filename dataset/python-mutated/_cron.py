from __future__ import annotations
import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any
from cron_descriptor import CasingTypeEnum, ExpressionDescriptor, FormatException, MissingFieldException
from croniter import CroniterBadCronError, CroniterBadDateError, croniter
from pendulum.tz.timezone import Timezone
from airflow.exceptions import AirflowTimetableInvalid
from airflow.utils.dates import cron_presets
from airflow.utils.timezone import convert_to_utc, make_aware, make_naive
if TYPE_CHECKING:
    from pendulum import DateTime

def _is_schedule_fixed(expression: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Figures out if the schedule has a fixed time (e.g. 3 AM every day).\n\n    :return: True if the schedule has a fixed time, False if not.\n\n    Detection is done by "peeking" the next two cron trigger time; if the\n    two times have the same minute and hour value, the schedule is fixed,\n    and we *don\'t* need to perform the DST fix.\n\n    This assumes DST happens on whole minute changes (e.g. 12:59 -> 12:00).\n    '
    cron = croniter(expression)
    next_a = cron.get_next(datetime.datetime)
    next_b = cron.get_next(datetime.datetime)
    return next_b.minute == next_a.minute and next_b.hour == next_a.hour

class CronMixin:
    """Mixin to provide interface to work with croniter."""

    def __init__(self, cron: str, timezone: str | Timezone) -> None:
        if False:
            print('Hello World!')
        self._expression = cron_presets.get(cron, cron)
        if isinstance(timezone, str):
            timezone = Timezone(timezone)
        self._timezone = timezone
        try:
            descriptor = ExpressionDescriptor(expression=self._expression, casing_type=CasingTypeEnum.Sentence, use_24hour_time_format=True)
            if len(croniter(self._expression).expanded) > 5:
                raise FormatException()
            interval_description: str = descriptor.get_description()
        except (CroniterBadCronError, FormatException, MissingFieldException):
            interval_description = ''
        self.description: str = interval_description

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        'Both expression and timezone should match.\n\n        This is only for testing purposes and should not be relied on otherwise.\n        '
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._expression == other._expression and self._timezone == other._timezone

    @property
    def summary(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._expression

    def validate(self) -> None:
        if False:
            return 10
        try:
            croniter(self._expression)
        except (CroniterBadCronError, CroniterBadDateError) as e:
            raise AirflowTimetableInvalid(str(e))

    @cached_property
    def _should_fix_dst(self) -> bool:
        if False:
            i = 10
            return i + 15
        return not _is_schedule_fixed(self._expression)

    def _get_next(self, current: DateTime) -> DateTime:
        if False:
            print('Hello World!')
        'Get the first schedule after specified time, with DST fixed.'
        naive = make_naive(current, self._timezone)
        cron = croniter(self._expression, start_time=naive)
        scheduled = cron.get_next(datetime.datetime)
        if not self._should_fix_dst:
            return convert_to_utc(make_aware(scheduled, self._timezone))
        delta = scheduled - naive
        return convert_to_utc(current.in_timezone(self._timezone) + delta)

    def _get_prev(self, current: DateTime) -> DateTime:
        if False:
            i = 10
            return i + 15
        'Get the first schedule before specified time, with DST fixed.'
        naive = make_naive(current, self._timezone)
        cron = croniter(self._expression, start_time=naive)
        scheduled = cron.get_prev(datetime.datetime)
        if not self._should_fix_dst:
            return convert_to_utc(make_aware(scheduled, self._timezone))
        delta = naive - scheduled
        return convert_to_utc(current.in_timezone(self._timezone) - delta)

    def _align_to_next(self, current: DateTime) -> DateTime:
        if False:
            while True:
                i = 10
        'Get the next scheduled time.\n\n        This is ``current + interval``, unless ``current`` falls right on the\n        interval boundary, when ``current`` is returned.\n        '
        next_time = self._get_next(current)
        if self._get_prev(next_time) != current:
            return next_time
        return current

    def _align_to_prev(self, current: DateTime) -> DateTime:
        if False:
            i = 10
            return i + 15
        'Get the prev scheduled time.\n\n        This is ``current - interval``, unless ``current`` falls right on the\n        interval boundary, when ``current`` is returned.\n        '
        prev_time = self._get_prev(current)
        if self._get_next(prev_time) != current:
            return prev_time
        return current