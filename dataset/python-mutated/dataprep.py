"""This module contains a Dataprep Job sensor."""
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.providers.google.cloud.hooks.dataprep import GoogleDataprepHook, JobGroupStatuses
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class DataprepJobGroupIsFinishedSensor(BaseSensorOperator):
    """
    Check the status of the Dataprep task to be finished.

    :param job_group_id: ID of the job group to check
    """
    template_fields: Sequence[str] = ('job_group_id',)

    def __init__(self, *, job_group_id: int | str, dataprep_conn_id: str='dataprep_default', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.job_group_id = job_group_id
        self.dataprep_conn_id = dataprep_conn_id

    def poke(self, context: Context) -> bool:
        if False:
            i = 10
            return i + 15
        hooks = GoogleDataprepHook(dataprep_conn_id=self.dataprep_conn_id)
        status = hooks.get_job_group_status(job_group_id=int(self.job_group_id))
        return status != JobGroupStatuses.IN_PROGRESS