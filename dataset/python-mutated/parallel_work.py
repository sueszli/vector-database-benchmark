import datetime
import time
from airflow.models import DAG
from airflow.operators.python import PythonOperator
default_args = {'start_date': datetime.datetime(2022, 4, 1)}
with DAG('parallel_work', default_args=default_args, schedule_interval=None) as dag:

    def do_work() -> None:
        if False:
            i = 10
            return i + 15
        import logging
        logging.info('Doing work')
        time.sleep(60)
        logging.info('Done!')
    for i in range(12):
        tasks = []
        for j in range(6):
            task = PythonOperator(task_id=f'work-{i}-{j}', python_callable=do_work)
            if tasks:
                tasks[-1] >> task
            tasks.append(task)