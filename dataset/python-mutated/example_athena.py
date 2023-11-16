from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.athena import AthenaOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3CreateObjectOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.sensors.athena import AthenaSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
sys_test_context_task = SystemTestContextBuilder().build()
DAG_ID = 'example_athena'
SAMPLE_DATA = '"Alice",20\n    "Bob",25\n    "Charlie",30\n    '
SAMPLE_FILENAME = 'airflow_sample.csv'

@task
def await_bucket(bucket_name):
    if False:
        print('Hello World!')
    client = boto3.client('s3')
    waiter = client.get_waiter('bucket_exists')
    waiter.wait(Bucket=bucket_name)

@task
def read_results_from_s3(bucket_name, query_execution_id):
    if False:
        i = 10
        return i + 15
    s3_hook = S3Hook()
    file_obj = s3_hook.get_conn().get_object(Bucket=bucket_name, Key=f'{query_execution_id}.csv')
    file_content = file_obj['Body'].read().decode('utf-8')
    print(file_content)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    s3_bucket = f'{env_id}-athena-bucket'
    athena_table = f'{env_id}_test_table'
    athena_database = f'{env_id}_default'
    query_create_database = f'CREATE DATABASE IF NOT EXISTS {athena_database}'
    query_create_table = f'CREATE EXTERNAL TABLE IF NOT EXISTS {athena_database}.{athena_table}\n        ( `name` string, `age` int )\n        ROW FORMAT SERDE "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"\n        WITH SERDEPROPERTIES ( "serialization.format" = ",", "field.delim" = "," )\n        LOCATION "s3://{s3_bucket}//{athena_table}"\n        TBLPROPERTIES ("has_encrypted_data"="false")\n        '
    query_read_table = f'SELECT * from {athena_database}.{athena_table}'
    query_drop_table = f'DROP TABLE IF EXISTS {athena_database}.{athena_table}'
    query_drop_database = f'DROP DATABASE IF EXISTS {athena_database}'
    create_s3_bucket = S3CreateBucketOperator(task_id='create_s3_bucket', bucket_name=s3_bucket)
    upload_sample_data = S3CreateObjectOperator(task_id='upload_sample_data', s3_bucket=s3_bucket, s3_key=f'{athena_table}/{SAMPLE_FILENAME}', data=SAMPLE_DATA, replace=True)
    create_database = AthenaOperator(task_id='create_database', query=query_create_database, database=athena_database, output_location=f's3://{s3_bucket}/', sleep_time=1)
    create_table = AthenaOperator(task_id='create_table', query=query_create_table, database=athena_database, output_location=f's3://{s3_bucket}/', sleep_time=1)
    read_table = AthenaOperator(task_id='read_table', query=query_read_table, database=athena_database, output_location=f's3://{s3_bucket}/')
    read_table.sleep_time = 1
    await_query = AthenaSensor(task_id='await_query', query_execution_id=read_table.output)
    drop_table = AthenaOperator(task_id='drop_table', query=query_drop_table, database=athena_database, output_location=f's3://{s3_bucket}/', trigger_rule=TriggerRule.ALL_DONE, sleep_time=1)
    drop_database = AthenaOperator(task_id='drop_database', query=query_drop_database, database=athena_database, output_location=f's3://{s3_bucket}/', trigger_rule=TriggerRule.ALL_DONE, sleep_time=1)
    delete_s3_bucket = S3DeleteBucketOperator(task_id='delete_s3_bucket', bucket_name=s3_bucket, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_s3_bucket, await_bucket(s3_bucket), upload_sample_data, create_database, create_table, read_table, await_query, read_results_from_s3(s3_bucket, read_table.output), drop_table, drop_database, delete_s3_bucket)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)