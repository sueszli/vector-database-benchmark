"""
This is an example DAG for the use of the SqliteOperator.
In this example, we create two tasks that execute in sequence.
The first task calls an sql command, defined in the SQLite operator,
which when triggered, is performed on the connected sqlite database.
The second task is similar but instead calls the SQL command from an external file.
"""
from __future__ import annotations
import os
from datetime import datetime
from airflow import DAG
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_sqlite'
with DAG(dag_id=DAG_ID, schedule='@daily', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    create_table_sqlite_task = SqliteOperator(task_id='create_table_sqlite', sql='\n        CREATE TABLE Customers (\n            customer_id INT PRIMARY KEY,\n            first_name TEXT,\n            last_name TEXT\n        );\n        ')

    @dag.task(task_id='insert_sqlite_task')
    def insert_sqlite_hook():
        if False:
            i = 10
            return i + 15
        sqlite_hook = SqliteHook()
        rows = [('James', '11'), ('James', '22'), ('James', '33')]
        target_fields = ['first_name', 'last_name']
        sqlite_hook.insert_rows(table='Customers', rows=rows, target_fields=target_fields)

    @dag.task(task_id='replace_sqlite_task')
    def replace_sqlite_hook():
        if False:
            while True:
                i = 10
        sqlite_hook = SqliteHook()
        rows = [('James', '11'), ('James', '22'), ('James', '33')]
        target_fields = ['first_name', 'last_name']
        sqlite_hook.insert_rows(table='Customers', rows=rows, target_fields=target_fields, replace=True)
    external_create_table_sqlite_task = SqliteOperator(task_id='create_table_sqlite_external_file', sql='create_table.sql')
    create_table_sqlite_task >> external_create_table_sqlite_task >> insert_sqlite_hook() >> replace_sqlite_hook()
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)