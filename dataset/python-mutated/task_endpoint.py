from __future__ import annotations
from operator import attrgetter
from typing import TYPE_CHECKING
from airflow.api_connexion import security
from airflow.api_connexion.exceptions import BadRequest, NotFound
from airflow.api_connexion.schemas.task_schema import TaskCollection, task_collection_schema, task_schema
from airflow.auth.managers.models.resource_details import DagAccessEntity
from airflow.exceptions import TaskNotFound
from airflow.utils.airflow_flask_app import get_airflow_app
if TYPE_CHECKING:
    from airflow import DAG
    from airflow.api_connexion.types import APIResponse

@security.requires_access_dag('GET', DagAccessEntity.TASK)
def get_task(*, dag_id: str, task_id: str) -> APIResponse:
    if False:
        print('Hello World!')
    'Get simplified representation of a task.'
    dag: DAG = get_airflow_app().dag_bag.get_dag(dag_id)
    if not dag:
        raise NotFound('DAG not found')
    try:
        task = dag.get_task(task_id=task_id)
    except TaskNotFound:
        raise NotFound('Task not found')
    return task_schema.dump(task)

@security.requires_access_dag('GET', DagAccessEntity.TASK)
def get_tasks(*, dag_id: str, order_by: str='task_id') -> APIResponse:
    if False:
        i = 10
        return i + 15
    'Get tasks for DAG.'
    dag: DAG = get_airflow_app().dag_bag.get_dag(dag_id)
    if not dag:
        raise NotFound('DAG not found')
    tasks = dag.tasks
    try:
        tasks = sorted(tasks, key=attrgetter(order_by.lstrip('-')), reverse=order_by[0:1] == '-')
    except AttributeError as err:
        raise BadRequest(detail=str(err))
    task_collection = TaskCollection(tasks=tasks, total_entries=len(tasks))
    return task_collection_schema.dump(task_collection)