from __future__ import annotations
import os
from datetime import datetime
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
sys_test_context_task = SystemTestContextBuilder().build()
DAG_ID = 'example_local_to_s3'
TEMP_FILE_PATH = '/tmp/sample-txt.txt'
SAMPLE_TEXT = 'This is some sample text.'

@task
def create_temp_file():
    if False:
        for i in range(10):
            print('nop')
    file = open(TEMP_FILE_PATH, 'w')
    file.write(SAMPLE_TEXT)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_temp_file():
    if False:
        i = 10
        return i + 15
    if os.path.exists(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    s3_bucket_name = f'{env_id}-bucket'
    s3_key = f'{env_id}/files/my-temp-file.txt'
    create_s3_bucket = S3CreateBucketOperator(task_id='create-s3-bucket', bucket_name=s3_bucket_name)
    create_local_to_s3_job = LocalFilesystemToS3Operator(task_id='create_local_to_s3_job', filename=TEMP_FILE_PATH, dest_key=s3_key, dest_bucket=s3_bucket_name, replace=True)
    delete_s3_bucket = S3DeleteBucketOperator(task_id='delete_s3_bucket', bucket_name=s3_bucket_name, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_temp_file(), create_s3_bucket, create_local_to_s3_job, delete_s3_bucket, delete_temp_file())
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)