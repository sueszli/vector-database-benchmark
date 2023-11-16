from __future__ import annotations
import os
from datetime import datetime
from airflow.decorators import task
from airflow.models import DAG
from airflow.providers.microsoft.azure.hooks.fileshare import AzureFileShareHook
NAME = 'myfileshare'
DIRECTORY = 'mydirectory'
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_fileshare'

@task
def create_fileshare():
    if False:
        return 10
    'Create a fileshare with directory'
    hook = AzureFileShareHook()
    hook.create_share(NAME)
    hook.create_directory(share_name=NAME, directory_name=DIRECTORY)
    exists = hook.check_for_directory(share_name=NAME, directory_name=DIRECTORY)
    if not exists:
        raise Exception

@task
def delete_fileshare():
    if False:
        while True:
            i = 10
    'Delete a fileshare'
    hook = AzureFileShareHook()
    hook.delete_share(NAME)
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False) as dag:
    create_fileshare() >> delete_fileshare()
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)