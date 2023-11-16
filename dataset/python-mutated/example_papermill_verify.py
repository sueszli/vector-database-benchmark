"""
This DAG will use Papermill to run the notebook "hello_world", based on the execution date
it will create an output notebook "out-<date>". All fields, including the keys in the parameters, are
templated.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
import scrapbook as sb
from airflow import DAG
from airflow.decorators import task
from airflow.lineage import AUTO
from airflow.providers.papermill.operators.papermill import PapermillOperator
START_DATE = datetime(2021, 1, 1)
SCHEDULE_INTERVAL = '0 0 * * *'
DAGRUN_TIMEOUT = timedelta(minutes=60)
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_papermill_operator_verify'

@task
def check_notebook(inlets, execution_date):
    if False:
        i = 10
        return i + 15
    '\n    Verify the message in the notebook\n    '
    notebook = sb.read_notebook(inlets[0].url)
    message = notebook.scraps['message']
    print(f'Message in notebook {message} for {execution_date}')
    if message.data != f'Ran from Airflow at {execution_date}!':
        return False
    return True
with DAG(dag_id='example_papermill_operator_verify', schedule=SCHEDULE_INTERVAL, start_date=START_DATE, dagrun_timeout=DAGRUN_TIMEOUT, catchup=False) as dag:
    run_this = PapermillOperator(task_id='run_example_notebook', input_nb=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'input_notebook.ipynb'), output_nb='/tmp/out-{{ execution_date }}.ipynb', parameters={'msgs': 'Ran from Airflow at {{ execution_date }}!'})
    run_this >> check_notebook(inlets=AUTO, execution_date='{{ execution_date }}')
from tests.system.utils import get_test_run
test_run = get_test_run(dag)