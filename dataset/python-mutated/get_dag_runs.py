"""DAG runs APIs."""
from __future__ import annotations
from typing import Any
from flask import url_for
from airflow.api.common.experimental import check_and_get_dag
from airflow.models import DagRun
from airflow.utils.state import DagRunState

def get_dag_runs(dag_id: str, state: str | None=None) -> list[dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of Dag Runs for a specific DAG ID.\n\n    :param dag_id: String identifier of a DAG\n    :param state: queued|running|success...\n    :return: List of DAG runs of a DAG with requested state,\n        or all runs if the state is not specified\n    '
    check_and_get_dag(dag_id=dag_id)
    dag_runs = []
    state = DagRunState(state.lower()) if state else None
    for run in DagRun.find(dag_id=dag_id, state=state):
        dag_runs.append({'id': run.id, 'run_id': run.run_id, 'state': run.state, 'dag_id': run.dag_id, 'execution_date': run.execution_date.isoformat(), 'start_date': (run.start_date or '') and run.start_date.isoformat(), 'dag_run_url': url_for('Airflow.graph', dag_id=run.dag_id, execution_date=run.execution_date)})
    return dag_runs