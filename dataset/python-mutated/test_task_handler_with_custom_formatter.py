from __future__ import annotations
import logging
import pytest
from airflow.config_templates.airflow_local_settings import DEFAULT_LOGGING_CONFIG
from airflow.models.dag import DAG
from airflow.models.taskinstance import TaskInstance
from airflow.operators.empty import EmptyOperator
from airflow.utils.log.logging_mixin import set_context
from airflow.utils.state import DagRunState
from airflow.utils.timezone import datetime
from airflow.utils.types import DagRunType
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_runs
pytestmark = pytest.mark.db_test
DEFAULT_DATE = datetime(2019, 1, 1)
TASK_HANDLER = 'task'
TASK_HANDLER_CLASS = 'airflow.utils.log.task_handler_with_custom_formatter.TaskHandlerWithCustomFormatter'
PREV_TASK_HANDLER = DEFAULT_LOGGING_CONFIG['handlers']['task']
DAG_ID = 'task_handler_with_custom_formatter_dag'
TASK_ID = 'task_handler_with_custom_formatter_task'

@pytest.fixture(scope='module', autouse=True)
def custom_task_log_handler_config():
    if False:
        print('Hello World!')
    DEFAULT_LOGGING_CONFIG['handlers']['task'] = {'class': TASK_HANDLER_CLASS, 'formatter': 'airflow', 'stream': 'sys.stdout'}
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
    logging.root.disabled = False
    yield
    DEFAULT_LOGGING_CONFIG['handlers']['task'] = PREV_TASK_HANDLER
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

@pytest.fixture()
def task_instance():
    if False:
        while True:
            i = 10
    dag = DAG(DAG_ID, start_date=DEFAULT_DATE)
    task = EmptyOperator(task_id=TASK_ID, dag=dag)
    dagrun = dag.create_dagrun(DagRunState.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.MANUAL)
    ti = TaskInstance(task=task, run_id=dagrun.run_id)
    ti.log.disabled = False
    yield ti
    clear_db_runs()

def assert_prefix(task_instance: TaskInstance, prefix: str) -> None:
    if False:
        i = 10
        return i + 15
    handler = next((h for h in task_instance.log.handlers if h.name == TASK_HANDLER), None)
    assert handler is not None, 'custom task log handler not set up correctly'
    assert handler.formatter is not None, 'custom task log formatter not set up correctly'
    expected_format = f'{prefix}:{handler.formatter._fmt}'
    set_context(task_instance.log, task_instance)
    assert expected_format == handler.formatter._fmt

def test_custom_formatter_default_format(task_instance):
    if False:
        i = 10
        return i + 15
    'The default format provides no prefix.'
    assert_prefix(task_instance, '')

@conf_vars({('logging', 'task_log_prefix_template'): '{{ti.dag_id }}-{{ ti.task_id }}'})
def test_custom_formatter_custom_format_not_affected_by_config(task_instance):
    if False:
        i = 10
        return i + 15
    assert_prefix(task_instance, f'{DAG_ID}-{TASK_ID}')