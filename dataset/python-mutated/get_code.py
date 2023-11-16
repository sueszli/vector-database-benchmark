"""Get code APIs."""
from __future__ import annotations
from deprecated import deprecated
from airflow.api.common.experimental import check_and_get_dag
from airflow.exceptions import AirflowException, DagCodeNotFound
from airflow.models.dagcode import DagCode

@deprecated(reason='Use DagCode().get_code_by_fileloc() instead', version='2.2.4')
def get_code(dag_id: str) -> str:
    if False:
        while True:
            i = 10
    'Return python code of a given dag_id.\n\n    :param dag_id: DAG id\n    :return: code of the DAG\n    '
    dag = check_and_get_dag(dag_id=dag_id)
    try:
        return DagCode.get_code_by_fileloc(dag.fileloc)
    except (OSError, DagCodeNotFound) as exception:
        error_message = f'Error {exception} while reading Dag id {dag_id} Code'
        raise AirflowException(error_message, exception)