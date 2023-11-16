"""Triggering DAG runs APIs."""
from __future__ import annotations
import json
from typing import TYPE_CHECKING
from airflow.exceptions import DagNotFound, DagRunAlreadyExists
from airflow.models import DagBag, DagModel, DagRun
from airflow.utils import timezone
from airflow.utils.state import DagRunState
from airflow.utils.types import DagRunType
if TYPE_CHECKING:
    from datetime import datetime

def _trigger_dag(dag_id: str, dag_bag: DagBag, run_id: str | None=None, conf: dict | str | None=None, execution_date: datetime | None=None, replace_microseconds: bool=True) -> list[DagRun | None]:
    if False:
        while True:
            i = 10
    'Triggers DAG run.\n\n    :param dag_id: DAG ID\n    :param dag_bag: DAG Bag model\n    :param run_id: ID of the dag_run\n    :param conf: configuration\n    :param execution_date: date of execution\n    :param replace_microseconds: whether microseconds should be zeroed\n    :return: list of triggered dags\n    '
    dag = dag_bag.get_dag(dag_id)
    if dag is None or dag_id not in dag_bag.dags:
        raise DagNotFound(f'Dag id {dag_id} not found')
    execution_date = execution_date or timezone.utcnow()
    if not timezone.is_localized(execution_date):
        raise ValueError('The execution_date should be localized')
    if replace_microseconds:
        execution_date = execution_date.replace(microsecond=0)
    if dag.default_args and 'start_date' in dag.default_args:
        min_dag_start_date = dag.default_args['start_date']
        if min_dag_start_date and execution_date < min_dag_start_date:
            raise ValueError(f"The execution_date [{execution_date.isoformat()}] should be >= start_date [{min_dag_start_date.isoformat()}] from DAG's default_args")
    logical_date = timezone.coerce_datetime(execution_date)
    data_interval = dag.timetable.infer_manual_data_interval(run_after=logical_date)
    run_id = run_id or dag.timetable.generate_run_id(run_type=DagRunType.MANUAL, logical_date=logical_date, data_interval=data_interval)
    dag_run = DagRun.find_duplicate(dag_id=dag_id, execution_date=execution_date, run_id=run_id)
    if dag_run:
        raise DagRunAlreadyExists(dag_run=dag_run, execution_date=execution_date, run_id=run_id)
    run_conf = None
    if conf:
        run_conf = conf if isinstance(conf, dict) else json.loads(conf)
    dag_runs = []
    dags_to_run = [dag, *dag.subdags]
    for _dag in dags_to_run:
        dag_run = _dag.create_dagrun(run_id=run_id, execution_date=execution_date, state=DagRunState.QUEUED, conf=run_conf, external_trigger=True, dag_hash=dag_bag.dags_hash.get(dag_id), data_interval=data_interval)
        dag_runs.append(dag_run)
    return dag_runs

def trigger_dag(dag_id: str, run_id: str | None=None, conf: dict | str | None=None, execution_date: datetime | None=None, replace_microseconds: bool=True) -> DagRun | None:
    if False:
        for i in range(10):
            print('nop')
    'Triggers execution of DAG specified by dag_id.\n\n    :param dag_id: DAG ID\n    :param run_id: ID of the dag_run\n    :param conf: configuration\n    :param execution_date: date of execution\n    :param replace_microseconds: whether microseconds should be zeroed\n    :return: first dag run triggered - even if more than one Dag Runs were triggered or None\n    '
    dag_model = DagModel.get_current(dag_id)
    if dag_model is None:
        raise DagNotFound(f'Dag id {dag_id} not found in DagModel')
    dagbag = DagBag(dag_folder=dag_model.fileloc, read_dags_from_db=True)
    triggers = _trigger_dag(dag_id=dag_id, dag_bag=dagbag, run_id=run_id, conf=conf, execution_date=execution_date, replace_microseconds=replace_microseconds)
    return triggers[0] if triggers else None