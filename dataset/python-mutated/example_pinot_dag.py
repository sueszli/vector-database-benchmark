"""Example DAG demonstrating the usage of the PinotAdminHook and PinotDbApiHook."""
from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.providers.apache.pinot.hooks.pinot import PinotAdminHook, PinotDbApiHook
with DAG(dag_id='example_pinot_hook', schedule=None, start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:

    @task
    def pinot_admin():
        if False:
            while True:
                i = 10
        PinotAdminHook(conn_id='pinot_admin_default', cmd_path='pinot-admin.sh', pinot_admin_system_exit=True)

    @task
    def pinot_dbi_api():
        if False:
            i = 10
            return i + 15
        PinotDbApiHook(task_id='run_example_pinot_script', pinot='ls /;', pinot_options='-x local')
    pinot_admin()
    pinot_dbi_api()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)