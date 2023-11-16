"""DAG run APIs."""
from __future__ import annotations
from typing import TYPE_CHECKING
from deprecated import deprecated
from airflow.api.common.experimental import check_and_get_dag, check_and_get_dagrun
if TYPE_CHECKING:
    from datetime import datetime

@deprecated(reason='Use DagRun().get_state() instead', version='2.2.4')
def get_dag_run_state(dag_id: str, execution_date: datetime) -> dict[str, str]:
    if False:
        print('Hello World!')
    'Return the Dag Run state identified by the given dag_id and execution_date.\n\n    :param dag_id: DAG id\n    :param execution_date: execution date\n    :return: Dictionary storing state of the object\n    '
    dag = check_and_get_dag(dag_id=dag_id)
    dagrun = check_and_get_dagrun(dag, execution_date)
    return {'state': dagrun.get_state()}