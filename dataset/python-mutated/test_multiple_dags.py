from __future__ import annotations
import datetime
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
args = {'owner': 'airflow', 'retries': 3, 'start_date': datetime.datetime(2022, 1, 1)}

def create_dag(suffix):
    if False:
        for i in range(10):
            print('nop')
    dag = DAG(dag_id=f'test_multiple_dags__{suffix}', default_args=args, schedule='0 0 * * *', dagrun_timeout=datetime.timedelta(minutes=60))
    with dag:
        BashOperator(task_id='test_task', bash_command='echo', dag=dag)
    return dag
globals()['dag_1'] = create_dag('dag_1')
globals()['dag_2'] = create_dag('dag_2')