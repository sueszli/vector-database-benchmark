from __future__ import annotations
import asyncio
import datetime
from typing import Any
from airflow.triggers.base import BaseTrigger, TriggerEvent
from airflow.utils import timezone

class DateTimeTrigger(BaseTrigger):
    """
    Trigger based on a datetime.

    A trigger that fires exactly once, at the given datetime, give or take
    a few seconds.

    The provided datetime MUST be in UTC.
    """

    def __init__(self, moment: datetime.datetime):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if not isinstance(moment, datetime.datetime):
            raise TypeError(f'Expected datetime.datetime type for moment. Got {type(moment)}')
        elif moment.tzinfo is None:
            raise ValueError('You cannot pass naive datetimes')
        else:
            self.moment = timezone.convert_to_utc(moment)

    def serialize(self) -> tuple[str, dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        return ('airflow.triggers.temporal.DateTimeTrigger', {'moment': self.moment})

    async def run(self):
        """
        Loop until the relevant time is met.

        We do have a two-phase delay to save some cycles, but sleeping is so
        cheap anyway that it's pretty loose. We also don't just sleep for
        "the number of seconds until the time" in case the system clock changes
        unexpectedly, or handles a DST change poorly.
        """
        self.log.info('trigger starting')
        for step in (3600, 60, 10):
            seconds_remaining = (self.moment - timezone.utcnow()).total_seconds()
            while seconds_remaining > 2 * step:
                self.log.info(f'{int(seconds_remaining)} seconds remaining; sleeping {step} seconds')
                await asyncio.sleep(step)
                seconds_remaining = (self.moment - timezone.utcnow()).total_seconds()
        while self.moment > timezone.utcnow():
            self.log.info('sleeping 1 second...')
            await asyncio.sleep(1)
        self.log.info('yielding event with payload %r', self.moment)
        yield TriggerEvent(self.moment)

class TimeDeltaTrigger(DateTimeTrigger):
    """
    Create DateTimeTriggers based on delays.

    Subclass to create DateTimeTriggers based on time delays rather
    than exact moments.

    While this is its own distinct class here, it will serialise to a
    DateTimeTrigger class, since they're operationally the same.
    """

    def __init__(self, delta: datetime.timedelta):
        if False:
            return 10
        super().__init__(moment=timezone.utcnow() + delta)