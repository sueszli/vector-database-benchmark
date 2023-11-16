from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.dag_parsing_context import _AIRFLOW_PARSING_CONTEXT_DAG_ID, _AIRFLOW_PARSING_CONTEXT_TASK_ID
from airflow.utils.timezone import datetime
if TYPE_CHECKING:
    from airflow.utils.context import Context

class DagWithParsingContext(EmptyOperator):

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        import os
        parsing_context_file = Path('/tmp/airflow_parsing_context')
        self.log.info('Executing')
        parsing_context = f'{_AIRFLOW_PARSING_CONTEXT_DAG_ID}={os.environ.get(_AIRFLOW_PARSING_CONTEXT_DAG_ID)}\n{_AIRFLOW_PARSING_CONTEXT_TASK_ID}={os.environ.get(_AIRFLOW_PARSING_CONTEXT_TASK_ID)}\n'
        parsing_context_file.write_text(parsing_context)
        self.log.info('Executed')
dag1 = DAG(dag_id='test_parsing_context', start_date=datetime(2015, 1, 1))
dag1_task1 = DagWithParsingContext(task_id='task1', dag=dag1, owner='airflow')