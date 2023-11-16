from __future__ import annotations
import boto3
import pendulum
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.glue_databrew import GlueDataBrewStartJobOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3CreateObjectOperator, S3DeleteBucketOperator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
DAG_ID = 'example_glue_databrew'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()
EXAMPLE_JSON = '{}'

@task
def create_dataset(dataset_name: str, bucket_name: str, object_key: str):
    if False:
        i = 10
        return i + 15
    client = boto3.client('databrew')
    client.create_dataset(Name=dataset_name, Format='JSON', FormatOptions={'Json': {'MultiLine': False}}, Input={'S3InputDefinition': {'Bucket': bucket_name, 'Key': object_key}})

@task
def create_job(dataset_name: str, job_name: str, bucket_output_name: str, object_output_key: str, role_arn: str):
    if False:
        while True:
            i = 10
    client = boto3.client('databrew')
    client.create_profile_job(DatasetName=dataset_name, Name=job_name, LogSubscription='ENABLE', OutputLocation={'Bucket': bucket_output_name, 'Key': object_output_key}, RoleArn=role_arn)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_dataset(dataset_name: str):
    if False:
        return 10
    client = boto3.client('databrew')
    client.delete_dataset(Name=dataset_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_job(job_name: str):
    if False:
        for i in range(10):
            print('nop')
    client = boto3.client('databrew')
    client.delete_job(Name=job_name)
with DAG(DAG_ID, schedule='@once', start_date=pendulum.datetime(2023, 1, 1, tz='UTC'), catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    role_arn = test_context[ROLE_ARN_KEY]
    bucket_name = f'{env_id}-bucket-databrew'
    output_bucket_name = f'{env_id}-output-bucket-databrew'
    file_name = 'data.json'
    dataset_name = f'{env_id}-dataset'
    job_name = f'{env_id}-databrew-job'
    create_bucket = S3CreateBucketOperator(task_id='create_bucket', bucket_name=bucket_name)
    create_output_bucket = S3CreateBucketOperator(task_id='create_output_bucket', bucket_name=output_bucket_name)
    upload_file = S3CreateObjectOperator(task_id='upload_file', s3_bucket=bucket_name, s3_key=file_name, data=EXAMPLE_JSON, replace=True)
    start_job = GlueDataBrewStartJobOperator(task_id='startjob', job_name=job_name, delay=15)
    delete_bucket = S3DeleteBucketOperator(task_id='delete_bucket', trigger_rule=TriggerRule.ALL_DONE, bucket_name=bucket_name, force_delete=True)
    delete_output_bucket = S3DeleteBucketOperator(task_id='delete_output_bucket', trigger_rule=TriggerRule.ALL_DONE, bucket_name=output_bucket_name, force_delete=True)
    chain(test_context, create_bucket, create_output_bucket, upload_file, create_dataset(dataset_name, bucket_name, file_name), create_job(dataset_name, job_name, output_bucket_name, 'output.json', role_arn), start_job, delete_job(job_name), delete_dataset(dataset_name), delete_bucket, delete_output_bucket)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)