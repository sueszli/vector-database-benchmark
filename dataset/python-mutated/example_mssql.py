"""
Example use of MsSql related operators.
"""
from __future__ import annotations
import os
from datetime import datetime
import pytest
from airflow import DAG
try:
    from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
    from airflow.providers.microsoft.mssql.operators.mssql import MsSqlOperator
except ImportError:
    pytest.skip('MSSQL provider not available', allow_module_level=True)
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_mssql'
with DAG(DAG_ID, schedule='@daily', start_date=datetime(2021, 10, 1), tags=['example'], catchup=False) as dag:
    create_table_mssql_task = MsSqlOperator(task_id='create_country_table', mssql_conn_id='airflow_mssql', sql='\n        CREATE TABLE Country (\n            country_id INT NOT NULL IDENTITY(1,1) PRIMARY KEY,\n            name TEXT,\n            continent TEXT\n        );\n        ', dag=dag)

    @dag.task(task_id='insert_mssql_task')
    def insert_mssql_hook():
        if False:
            for i in range(10):
                print('nop')
        mssql_hook = MsSqlHook(mssql_conn_id='airflow_mssql', schema='airflow')
        rows = [('India', 'Asia'), ('Germany', 'Europe'), ('Argentina', 'South America'), ('Ghana', 'Africa'), ('Japan', 'Asia'), ('Namibia', 'Africa')]
        target_fields = ['name', 'continent']
        mssql_hook.insert_rows(table='Country', rows=rows, target_fields=target_fields)
    create_table_mssql_from_external_file = MsSqlOperator(task_id='create_table_from_external_file', mssql_conn_id='airflow_mssql', sql='create_table.sql', dag=dag)
    populate_user_table = MsSqlOperator(task_id='populate_user_table', mssql_conn_id='airflow_mssql', sql="\n                INSERT INTO Users (username, description)\n                VALUES ( 'Danny', 'Musician');\n                INSERT INTO Users (username, description)\n                VALUES ( 'Simone', 'Chef');\n                INSERT INTO Users (username, description)\n                VALUES ( 'Lily', 'Florist');\n                INSERT INTO Users (username, description)\n                VALUES ( 'Tim', 'Pet shop owner');\n                ")
    get_all_countries = MsSqlOperator(task_id='get_all_countries', mssql_conn_id='airflow_mssql', sql='SELECT * FROM Country;')
    get_all_description = MsSqlOperator(task_id='get_all_description', mssql_conn_id='airflow_mssql', sql='SELECT description FROM Users;')
    get_countries_from_continent = MsSqlOperator(task_id='get_countries_from_continent', mssql_conn_id='airflow_mssql', sql="SELECT * FROM Country where {{ params.column }}='{{ params.value }}';", params={'column': 'CONVERT(VARCHAR, continent)', 'value': 'Asia'})
    create_table_mssql_task >> insert_mssql_hook() >> create_table_mssql_from_external_file >> populate_user_table >> get_all_countries >> get_all_description >> get_countries_from_continent
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)