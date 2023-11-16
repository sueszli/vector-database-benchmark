from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.datasync import DataSyncOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3DeleteBucketOperator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_datasync'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()

def get_s3_bucket_arn(bucket_name):
    if False:
        while True:
            i = 10
    return f'arn:aws:s3:::{bucket_name}'

def create_location(bucket_name, role_arn):
    if False:
        return 10
    client = boto3.client('datasync')
    response = client.create_location_s3(Subdirectory='test', S3BucketArn=get_s3_bucket_arn(bucket_name), S3Config={'BucketAccessRoleArn': role_arn})
    return response['LocationArn']

@task
def create_source_location(bucket_source, role_arn):
    if False:
        for i in range(10):
            print('nop')
    return create_location(bucket_source, role_arn)

@task
def create_destination_location(bucket_destination, role_arn):
    if False:
        while True:
            i = 10
    return create_location(bucket_destination, role_arn)

@task
def create_task(**kwargs):
    if False:
        return 10
    client = boto3.client('datasync')
    response = client.create_task(SourceLocationArn=kwargs['ti'].xcom_pull('create_source_location'), DestinationLocationArn=kwargs['ti'].xcom_pull('create_destination_location'))
    return response['TaskArn']

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_task(task_arn):
    if False:
        return 10
    client = boto3.client('datasync')
    client.delete_task(TaskArn=task_arn)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_task_created_by_operator(**kwargs):
    if False:
        i = 10
        return i + 15
    client = boto3.client('datasync')
    client.delete_task(TaskArn=kwargs['ti'].xcom_pull('create_and_execute_task')['TaskArn'])

@task(trigger_rule=TriggerRule.ALL_DONE)
def list_locations(bucket_source, bucket_destination):
    if False:
        i = 10
        return i + 15
    client = boto3.client('datasync')
    return client.list_locations(Filters=[{'Name': 'LocationUri', 'Values': [f's3://{bucket_source}/test/', f's3://{bucket_destination}/test/', f's3://{bucket_source}/test_create/', f's3://{bucket_destination}/test_create/'], 'Operator': 'In'}])

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_locations(locations):
    if False:
        while True:
            i = 10
    client = boto3.client('datasync')
    for location in locations['Locations']:
        client.delete_location(LocationArn=location['LocationArn'])
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example']) as dag:
    test_context = sys_test_context_task()
    s3_bucket_source: str = f'{test_context[ENV_ID_KEY]}-datasync-bucket-source'
    s3_bucket_destination: str = f'{test_context[ENV_ID_KEY]}-datasync-bucket-destination'
    create_s3_bucket_source = S3CreateBucketOperator(task_id='create_s3_bucket_source', bucket_name=s3_bucket_source)
    create_s3_bucket_destination = S3CreateBucketOperator(task_id='create_s3_bucket_destination', bucket_name=s3_bucket_destination)
    source_location = create_source_location(s3_bucket_source, test_context[ROLE_ARN_KEY])
    destination_location = create_destination_location(s3_bucket_destination, test_context[ROLE_ARN_KEY])
    created_task_arn = create_task()
    execute_task_by_arn = DataSyncOperator(task_id='execute_task_by_arn', task_arn=created_task_arn)
    execute_task_by_arn.wait_for_completion = False
    execute_task_by_locations = DataSyncOperator(task_id='execute_task_by_locations', source_location_uri=f's3://{s3_bucket_source}/test', destination_location_uri=f's3://{s3_bucket_destination}/test', task_execution_kwargs={'Includes': [{'FilterType': 'SIMPLE_PATTERN', 'Value': '/test/subdir'}]})
    execute_task_by_locations.wait_for_completion = False
    create_and_execute_task = DataSyncOperator(task_id='create_and_execute_task', source_location_uri=f's3://{s3_bucket_source}/test_create', destination_location_uri=f's3://{s3_bucket_destination}/test_create', create_task_kwargs={'Name': 'Created by Airflow'}, create_source_location_kwargs={'Subdirectory': 'test_create', 'S3BucketArn': get_s3_bucket_arn(s3_bucket_source), 'S3Config': {'BucketAccessRoleArn': test_context[ROLE_ARN_KEY]}}, create_destination_location_kwargs={'Subdirectory': 'test_create', 'S3BucketArn': get_s3_bucket_arn(s3_bucket_destination), 'S3Config': {'BucketAccessRoleArn': test_context[ROLE_ARN_KEY]}}, delete_task_after_execution=False)
    create_and_execute_task.wait_for_completion = False
    locations_task = list_locations(s3_bucket_source, s3_bucket_destination)
    delete_locations_task = delete_locations(locations_task)
    delete_s3_bucket_source = S3DeleteBucketOperator(task_id='delete_s3_bucket_source', bucket_name=s3_bucket_source, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    delete_s3_bucket_destination = S3DeleteBucketOperator(task_id='delete_s3_bucket_destination', bucket_name=s3_bucket_destination, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_s3_bucket_source, create_s3_bucket_destination, source_location, destination_location, created_task_arn, execute_task_by_arn, execute_task_by_locations, create_and_execute_task, delete_task(created_task_arn), delete_task_created_by_operator(), locations_task, delete_locations_task, delete_s3_bucket_source, delete_s3_bucket_destination)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)