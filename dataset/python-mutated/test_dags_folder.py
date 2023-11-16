from __future__ import annotations
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
with DAG(dag_id='test_dags_folder', schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False) as dag:

    @task(task_id='task')
    def return_file_path():
        if False:
            while True:
                i = 10
        'Print the Airflow context and ds variable from the context.'
        print(f'dag file location: {__file__}')
        return __file__
    return_file_path()