from __future__ import annotations
import time
from typing import TYPE_CHECKING
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.timezone import datetime
if TYPE_CHECKING:
    from airflow.utils.context import Context

class DummyWithOnKill(EmptyOperator):

    def execute(self, context: Context):
        if False:
            print('Hello World!')
        import os
        self.log.info('Signalling that I am running')
        with open('/tmp/airflow_on_kill_running', 'w') as f:
            f.write('ON_KILL_RUNNING')
        self.log.info('Signalled')
        if not os.fork():
            os.system('sleep 10')
        time.sleep(10)

    def on_kill(self):
        if False:
            return 10
        self.log.info('Executing on_kill')
        with open('/tmp/airflow_on_kill_killed', 'w') as f:
            f.write('ON_KILL_TEST')
        self.log.info('Executed on_kill')
dag1 = DAG(dag_id='test_on_kill', start_date=datetime(2015, 1, 1))
dag1_task1 = DummyWithOnKill(task_id='task1', dag=dag1, owner='airflow')