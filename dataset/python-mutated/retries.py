from datetime import datetime, timedelta
from random import random
from airflow.models import DAG
from airflow.operators.python import PythonOperator
FAILURE_PROBABILITY = 0.5
RETRIES = 1
default_args = {'start_date': datetime(2022, 4, 1), 'retries': RETRIES, 'retry_delay': timedelta(seconds=1)}
with DAG('retries', default_args=default_args, schedule_interval=None) as dag:

    def api_call(parameter: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        import logging
        if random() < FAILURE_PROBABILITY:
            logging.error('Error: Simulating API flakiness')
            raise RuntimeError('Error: Simulating API flakiness')
        logging.info(f'Calling API with parameter {parameter}')
    task1 = PythonOperator(task_id='task1', python_callable=api_call, op_kwargs={'parameter': 1})
    task2 = PythonOperator(task_id='task2', python_callable=api_call, op_kwargs={'parameter': 2})
    task3 = PythonOperator(task_id='task3', python_callable=api_call, op_kwargs={'parameter': 3})
    task4 = PythonOperator(task_id='task4', python_callable=api_call, op_kwargs={'parameter': 4})
    task1 >> [task2, task3] >> task4