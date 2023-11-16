from __future__ import annotations
import os
from datetime import datetime
from airflow.decorators import task
from airflow.models import DAG
from airflow.providers.sftp.sensors.sftp import SFTPSensor
from airflow.providers.ssh.operators.ssh import SSHOperator
SFTP_DIRECTORY = os.environ.get('SFTP_DIRECTORY', 'example-empty-directory').rstrip('/') + '/'
FULL_FILE_PATH = f'{SFTP_DIRECTORY}example_test_sftp_sensor_decory_file.txt'
SFTP_DEFAULT_CONNECTION = 'sftp_default'

@task.python
def sleep_function():
    if False:
        return 10
    import time
    time.sleep(60)
with DAG('example_sftp_sensor', schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'sftp']) as dag:

    @task.sftp_sensor(task_id='sftp_sensor', path=FULL_FILE_PATH, poke_interval=10)
    def sftp_sensor_decorator():
        if False:
            while True:
                i = 10
        print('Files were successfully found!')
        return 'done'
    remove_file_task_start = SSHOperator(task_id='remove_file_start', command=f'rm {FULL_FILE_PATH} || true', ssh_conn_id=SFTP_DEFAULT_CONNECTION)
    remove_file_task_end = SSHOperator(task_id='remove_file_end', command=f'rm {FULL_FILE_PATH} || true', ssh_conn_id=SFTP_DEFAULT_CONNECTION)
    create_decoy_file_task = SSHOperator(task_id='create_file', command=f'touch {FULL_FILE_PATH}', ssh_conn_id=SFTP_DEFAULT_CONNECTION)
    sleep_task = sleep_function()
    sftp_with_sensor = sftp_sensor_decorator()
    sftp_with_operator = SFTPSensor(task_id='sftp_operator', path=FULL_FILE_PATH, poke_interval=10)
    remove_file_task_start >> sleep_task >> create_decoy_file_task
    remove_file_task_start >> [sftp_with_operator, sftp_with_sensor] >> remove_file_task_end
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)