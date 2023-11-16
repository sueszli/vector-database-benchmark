from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from airflow.exceptions import AirflowNotFoundException
from airflow.providers.amazon.aws.hooks.rds import RdsHook
from airflow.providers.amazon.aws.utils.rds import RdsDbType
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class RdsBaseSensor(BaseSensorOperator):
    """Base operator that implements common functions for all sensors."""
    ui_color = '#ddbb77'
    ui_fgcolor = '#ffffff'

    def __init__(self, *args, aws_conn_id: str='aws_conn_id', hook_params: dict | None=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.hook_params = hook_params or {}
        self.aws_conn_id = aws_conn_id
        self.target_statuses: list[str] = []
        super().__init__(*args, **kwargs)

    @cached_property
    def hook(self):
        if False:
            for i in range(10):
                print('nop')
        return RdsHook(aws_conn_id=self.aws_conn_id, **self.hook_params)

class RdsSnapshotExistenceSensor(RdsBaseSensor):
    """
    Waits for RDS snapshot with a specific status.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:RdsSnapshotExistenceSensor`

    :param db_type: Type of the DB - either "instance" or "cluster"
    :param db_snapshot_identifier: The identifier for the DB snapshot
    :param target_statuses: Target status of snapshot
    """
    template_fields: Sequence[str] = ('db_snapshot_identifier', 'target_statuses')

    def __init__(self, *, db_type: str, db_snapshot_identifier: str, target_statuses: list[str] | None=None, aws_conn_id: str='aws_conn_id', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(aws_conn_id=aws_conn_id, **kwargs)
        self.db_type = RdsDbType(db_type)
        self.db_snapshot_identifier = db_snapshot_identifier
        self.target_statuses = target_statuses or ['available']

    def poke(self, context: Context):
        if False:
            while True:
                i = 10
        self.log.info('Poking for statuses : %s\nfor snapshot %s', self.target_statuses, self.db_snapshot_identifier)
        try:
            if self.db_type.value == 'instance':
                state = self.hook.get_db_snapshot_state(self.db_snapshot_identifier)
            else:
                state = self.hook.get_db_cluster_snapshot_state(self.db_snapshot_identifier)
        except AirflowNotFoundException:
            return False
        return state in self.target_statuses

class RdsExportTaskExistenceSensor(RdsBaseSensor):
    """
    Waits for RDS export task with a specific status.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:RdsExportTaskExistenceSensor`

    :param export_task_identifier: A unique identifier for the snapshot export task.
    :param target_statuses: Target status of export task
    """
    template_fields: Sequence[str] = ('export_task_identifier', 'target_statuses')

    def __init__(self, *, export_task_identifier: str, target_statuses: list[str] | None=None, aws_conn_id: str='aws_default', **kwargs):
        if False:
            return 10
        super().__init__(aws_conn_id=aws_conn_id, **kwargs)
        self.export_task_identifier = export_task_identifier
        self.target_statuses = target_statuses or ['starting', 'in_progress', 'complete', 'canceling', 'canceled']

    def poke(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Poking for statuses : %s\nfor export task %s', self.target_statuses, self.export_task_identifier)
        try:
            state = self.hook.get_export_task_state(self.export_task_identifier)
        except AirflowNotFoundException:
            return False
        return state in self.target_statuses

class RdsDbSensor(RdsBaseSensor):
    """
    Waits for an RDS instance or cluster to enter one of a number of states.

    .. seealso::
        For more information on how to use this sensor, take a look at the guide:
        :ref:`howto/sensor:RdsDbSensor`

    :param db_type: Type of the DB - either "instance" or "cluster" (default: 'instance')
    :param db_identifier: The AWS identifier for the DB
    :param target_statuses: Target status of DB
    """
    template_fields: Sequence[str] = ('db_identifier', 'db_type', 'target_statuses')

    def __init__(self, *, db_identifier: str, db_type: RdsDbType | str=RdsDbType.INSTANCE, target_statuses: list[str] | None=None, aws_conn_id: str='aws_default', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(aws_conn_id=aws_conn_id, **kwargs)
        self.db_identifier = db_identifier
        self.target_statuses = target_statuses or ['available']
        self.db_type = db_type

    def poke(self, context: Context):
        if False:
            print('Hello World!')
        db_type = RdsDbType(self.db_type)
        self.log.info('Poking for statuses : %s\nfor db instance %s', self.target_statuses, self.db_identifier)
        try:
            if db_type == RdsDbType.INSTANCE:
                state = self.hook.get_db_instance_state(self.db_identifier)
            else:
                state = self.hook.get_db_cluster_state(self.db_identifier)
        except AirflowNotFoundException:
            return False
        return state in self.target_statuses
__all__ = ['RdsExportTaskExistenceSensor', 'RdsDbSensor', 'RdsSnapshotExistenceSensor']