from __future__ import annotations
import datetime
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
dag = DAG(dag_id='test_dag_xcom_openlineage', default_args={'owner': 'airflow', 'retries': 3, 'start_date': datetime.datetime(2022, 1, 1)}, schedule='0 0 * * *', dagrun_timeout=datetime.timedelta(minutes=60))

def push_and_pull(ti, **kwargs):
    if False:
        return 10
    ti.xcom_push(key='pushed_key', value='asdf')
    ti.xcom_pull(key='pushed_key')
task = PythonOperator(task_id='push_and_pull', python_callable=push_and_pull, dag=dag)
if __name__ == '__main__':
    dag.cli()