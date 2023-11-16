from __future__ import annotations
from typing import TYPE_CHECKING
from celery.app import control
from airflow.exceptions import AirflowSkipException
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class CeleryQueueSensor(BaseSensorOperator):
    """
    Waits for a Celery queue to be empty.

    By default, in order to be considered empty, the queue must not have
    any tasks in the ``reserved``, ``scheduled`` or ``active`` states.

    :param celery_queue: The name of the Celery queue to wait for.
    :param target_task_id: Task id for checking
    """

    def __init__(self, *, celery_queue: str, target_task_id: str | None=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.celery_queue = celery_queue
        self.target_task_id = target_task_id

    def _check_task_id(self, context: Context) -> bool:
        if False:
            while True:
                i = 10
        "\n        Get the Celery result from the Airflow task ID and return True if the result has finished execution.\n\n        :param context: Airflow's execution context\n        :return: True if task has been executed, otherwise False\n        "
        ti = context['ti']
        celery_result = ti.xcom_pull(task_ids=self.target_task_id)
        return celery_result.ready()

    def poke(self, context: Context) -> bool:
        if False:
            return 10
        if self.target_task_id:
            return self._check_task_id(context)
        inspect_result = control.Inspect()
        reserved = inspect_result.reserved()
        scheduled = inspect_result.scheduled()
        active = inspect_result.active()
        try:
            reserved = len(reserved[self.celery_queue])
            scheduled = len(scheduled[self.celery_queue])
            active = len(active[self.celery_queue])
            self.log.info('Checking if celery queue %s is empty.', self.celery_queue)
            return reserved == 0 and scheduled == 0 and (active == 0)
        except KeyError:
            message = f'Could not locate Celery queue {self.celery_queue}'
            if self.soft_fail:
                raise AirflowSkipException(message)
            raise KeyError(message)
        except Exception as err:
            if self.soft_fail:
                raise AirflowSkipException from err
            raise