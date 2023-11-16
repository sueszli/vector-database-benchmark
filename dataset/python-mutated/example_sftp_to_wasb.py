from __future__ import annotations
import os
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.providers.microsoft.azure.operators.wasb_delete_blob import WasbDeleteBlobOperator
from airflow.providers.microsoft.azure.transfers.sftp_to_wasb import SFTPToWasbOperator
from airflow.providers.sftp.hooks.sftp import SFTPHook
from airflow.providers.sftp.operators.sftp import SFTPOperator
AZURE_CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME', 'airflow')
BLOB_PREFIX = os.environ.get('AZURE_BLOB_PREFIX', 'airflow')
SFTP_SRC_PATH = os.environ.get('SFTP_SRC_PATH', '/sftp')
LOCAL_FILE_PATH = os.environ.get('LOCAL_SRC_PATH', '/tmp')
SAMPLE_FILENAME = os.environ.get('SFTP_SAMPLE_FILENAME', 'sftp_to_wasb_test.txt')
FILE_COMPLETE_PATH = os.path.join(LOCAL_FILE_PATH, SAMPLE_FILENAME)
SFTP_FILE_COMPLETE_PATH = os.path.join(SFTP_SRC_PATH, SAMPLE_FILENAME)
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_sftp_to_wasb'

@task
def delete_sftp_file():
    if False:
        return 10
    'Delete a file at SFTP SERVER'
    SFTPHook().delete_file(SFTP_FILE_COMPLETE_PATH)
with DAG(DAG_ID, schedule=None, catchup=False, start_date=datetime(2021, 1, 1)) as dag:
    transfer_files_to_sftp_step = SFTPOperator(task_id='transfer_files_from_local_to_sftp', local_filepath=FILE_COMPLETE_PATH, remote_filepath=SFTP_FILE_COMPLETE_PATH)
    transfer_files_to_azure = SFTPToWasbOperator(task_id='transfer_files_from_sftp_to_wasb', sftp_source_path=SFTP_SRC_PATH, container_name=AZURE_CONTAINER_NAME, blob_prefix=BLOB_PREFIX)
    delete_blob_file_step = WasbDeleteBlobOperator(task_id='delete_blob_files', container_name=AZURE_CONTAINER_NAME, blob_name=BLOB_PREFIX + SAMPLE_FILENAME)
    transfer_files_to_sftp_step >> transfer_files_to_azure >> delete_blob_file_step >> delete_sftp_file()
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)