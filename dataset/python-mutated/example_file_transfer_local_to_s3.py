from __future__ import annotations
import os
import uuid
from datetime import datetime
from typing import cast
from airflow import DAG
from airflow.decorators import task
from airflow.io.store.path import ObjectStoragePath
from airflow.providers.common.io.operators.file_transfer import FileTransferOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_file_transfer_local_to_s3'
SAMPLE_TEXT = 'This is some sample text.'
TEMP_FILE_PATH = ObjectStoragePath('file:///tmp')
AWS_BUCKET_NAME = f'bucket-aws-{DAG_ID}-{ENV_ID}'.replace('_', '-')
AWS_BUCKET = ObjectStoragePath(f's3://{AWS_BUCKET_NAME}')
AWS_FILE_PATH = AWS_BUCKET

@task
def create_temp_file() -> ObjectStoragePath:
    if False:
        while True:
            i = 10
    path = ObjectStoragePath(TEMP_FILE_PATH / str(uuid.uuid4()))
    with path.open('w') as file:
        file.write(SAMPLE_TEXT)
    return path

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_temp_file(path: ObjectStoragePath):
    if False:
        i = 10
        return i + 15
    path.unlink()

@task
def remove_bucket():
    if False:
        i = 10
        return i + 15
    AWS_BUCKET.unlink(recursive=True)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    temp_file = create_temp_file()
    temp_file_path = cast(ObjectStoragePath, temp_file)
    transfer = FileTransferOperator(src=temp_file_path, dst=AWS_BUCKET, task_id='transfer')
    temp_file >> transfer >> remove_bucket() >> delete_temp_file(temp_file_path)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)