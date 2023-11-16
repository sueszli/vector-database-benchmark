from __future__ import annotations
import datetime
from typing import TYPE_CHECKING
from airflow.sensors.base import BaseSensorOperator
from airflow.triggers.temporal import DateTimeTrigger
from airflow.utils import timezone
if TYPE_CHECKING:
    from airflow.utils.context import Context

class TimeSensor(BaseSensorOperator):
    """
    Waits until the specified time of the day.

    :param target_time: time after which the job succeeds

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/operator:TimeSensor`

    """

    def __init__(self, *, target_time, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.target_time = target_time

    def poke(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Checking if the time (%s) has come', self.target_time)
        return timezone.make_naive(timezone.utcnow(), self.dag.timezone).time() > self.target_time

class TimeSensorAsync(BaseSensorOperator):
    """
    Waits until the specified time of the day.

    This frees up a worker slot while it is waiting.

    :param target_time: time after which the job succeeds

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/operator:TimeSensorAsync`
    """

    def __init__(self, *, target_time, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.target_time = target_time
        aware_time = timezone.coerce_datetime(datetime.datetime.combine(datetime.datetime.today(), self.target_time, self.dag.timezone))
        self.target_datetime = timezone.convert_to_utc(aware_time)

    def execute(self, context: Context):
        if False:
            return 10
        trigger = DateTimeTrigger(moment=self.target_datetime)
        self.defer(trigger=trigger, method_name='execute_complete')

    def execute_complete(self, context, event=None):
        if False:
            print('Hello World!')
        'Execute when the trigger fires - returns immediately.'
        return None