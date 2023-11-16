"""Task instance APIs."""
from __future__ import annotations
from typing import TYPE_CHECKING
from deprecated import deprecated
from airflow.api.common.experimental import check_and_get_dag, check_and_get_dagrun
from airflow.exceptions import TaskInstanceNotFound
from airflow.models import TaskInstance
if TYPE_CHECKING:
    from datetime import datetime

@deprecated(version='2.2.4', reason='Use DagRun.get_task_instance instead')
def get_task_instance(dag_id: str, task_id: str, execution_date: datetime) -> TaskInstance:
    if False:
        i = 10
        return i + 15
    'Return the task instance identified by the given dag_id, task_id and execution_date.'
    dag = check_and_get_dag(dag_id, task_id)
    dagrun = check_and_get_dagrun(dag=dag, execution_date=execution_date)
    task_instance = dagrun.get_task_instance(task_id)
    if not task_instance:
        error_message = f'Task {task_id} instance for date {execution_date} not found'
        raise TaskInstanceNotFound(error_message)
    if isinstance(task_instance, TaskInstance):
        return task_instance
    raise ValueError('not a TaskInstance')