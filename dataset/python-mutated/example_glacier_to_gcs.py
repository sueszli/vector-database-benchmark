from __future__ import annotations
from datetime import datetime
import boto3
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.python import task
from airflow.providers.amazon.aws.operators.glacier import GlacierCreateJobOperator, GlacierUploadArchiveOperator
from airflow.providers.amazon.aws.sensors.glacier import GlacierJobOperationSensor
from airflow.providers.amazon.aws.transfers.glacier_to_gcs import GlacierToGCSOperator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
sys_test_context_task = SystemTestContextBuilder().build()
DAG_ID = 'example_glacier_to_gcs'

@task
def create_vault(vault_name):
    if False:
        print('Hello World!')
    boto3.client('glacier').create_vault(vaultName=vault_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_vault(vault_name):
    if False:
        while True:
            i = 10
    boto3.client('glacier').delete_vault(vaultName=vault_name)
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    vault_name = f'{env_id}-vault'
    gcs_bucket_name = f'{env_id}-bucket'
    gcs_object_name = f'{env_id}-object'
    create_glacier_job = GlacierCreateJobOperator(task_id='create_glacier_job', vault_name=vault_name)
    JOB_ID = '{{ task_instance.xcom_pull("create_glacier_job")["jobId"] }}'
    wait_for_operation_complete = GlacierJobOperationSensor(vault_name=vault_name, job_id=JOB_ID, task_id='wait_for_operation_complete')
    upload_archive_to_glacier = GlacierUploadArchiveOperator(task_id='upload_data_to_glacier', vault_name=vault_name, body=b'Test Data')
    transfer_archive_to_gcs = GlacierToGCSOperator(task_id='transfer_archive_to_gcs', vault_name=vault_name, bucket_name=gcs_bucket_name, object_name=gcs_object_name, gzip=False, chunk_size=1024)
    chain(test_context, create_vault(vault_name), create_glacier_job, wait_for_operation_complete, upload_archive_to_glacier, transfer_archive_to_gcs, delete_vault(vault_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)