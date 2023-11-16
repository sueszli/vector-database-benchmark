from __future__ import annotations
import logging
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.timezone import datetime
logger = logging.getLogger(__name__)

def test_logging_fn(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Tests DAG logging.\n\n    :param kwargs:\n    '
    logger.info('Log from DAG Logger')
    kwargs['ti'].log.info('Log from TI Logger')
    print('Log from Print statement')
dag = DAG(dag_id='test_logging_dag', schedule=None, start_date=datetime(2016, 1, 1))
PythonOperator(task_id='test_task', python_callable=test_logging_fn, dag=dag)