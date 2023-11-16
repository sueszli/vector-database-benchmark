"""
Example Airflow DAG for Elasticsearch Query.
"""
from __future__ import annotations
import os
from datetime import datetime
from airflow import models
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.providers.elasticsearch.hooks.elasticsearch import ElasticsearchPythonHook, ElasticsearchSQLHook
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'elasticsearch_dag'
CONN_ID = 'elasticsearch_default'

@task(task_id='es_print_tables')
def show_tables():
    if False:
        while True:
            i = 10
    '\n    show_tables queries elasticsearch to list available tables\n    '
    es = ElasticsearchSQLHook(elasticsearch_conn_id=CONN_ID)
    with es.get_conn() as es_conn:
        tables = es_conn.execute('SHOW TABLES')
        for (table, *_) in tables:
            print(f'table: {table}')
    return True

def use_elasticsearch_hook():
    if False:
        print('Hello World!')
    '\n    Use ElasticSearchPythonHook to print results from a local Elasticsearch\n    '
    es_hosts = ['http://localhost:9200']
    es_hook = ElasticsearchPythonHook(hosts=es_hosts)
    query = {'query': {'match_all': {}}}
    result = es_hook.search(query=query)
    print(result)
    return True
with models.DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'elasticsearch']) as dag:
    execute_query = show_tables()
    execute_query
    es_python_test = PythonOperator(task_id='print_data_from_elasticsearch', python_callable=use_elasticsearch_hook)
from tests.system.utils import get_test_run
test_run = get_test_run(dag)