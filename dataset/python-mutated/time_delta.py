from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.exceptions import AirflowSkipException
from airflow.sensors.base import BaseSensorOperator
from airflow.triggers.temporal import DateTimeTrigger
from airflow.utils import timezone
if TYPE_CHECKING:
    from airflow.utils.context import Context

class TimeDeltaSensor(BaseSensorOperator):
    """
    Waits for a timedelta after the run's data interval.

    :param delta: time length to wait after the data interval before succeeding.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/operator:TimeDeltaSensor`


    """

    def __init__(self, *, delta, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.delta = delta

    def poke(self, context: Context):
        if False:
            while True:
                i = 10
        target_dttm = context['data_interval_end']
        target_dttm += self.delta
        self.log.info('Checking if the time (%s) has come', target_dttm)
        return timezone.utcnow() > target_dttm

class TimeDeltaSensorAsync(TimeDeltaSensor):
    """
    A deferrable drop-in replacement for TimeDeltaSensor.

    Will defers itself to avoid taking up a worker slot while it is waiting.

    :param delta: time length to wait after the data interval before succeeding.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/operator:TimeDeltaSensorAsync`

    """

    def execute(self, context: Context):
        if False:
            while True:
                i = 10
        target_dttm = context['data_interval_end']
        target_dttm += self.delta
        try:
            trigger = DateTimeTrigger(moment=target_dttm)
        except (TypeError, ValueError) as e:
            if self.soft_fail:
                raise AirflowSkipException('Skipping due to soft_fail is set to True.') from e
            raise
        self.defer(trigger=trigger, method_name='execute_complete')

    def execute_complete(self, context, event=None):
        if False:
            while True:
                i = 10
        'Execute for when the trigger fires - return immediately.'
        return None