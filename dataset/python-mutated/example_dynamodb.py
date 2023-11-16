from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.sensors.dynamodb import DynamoDBValueSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_dynamodbvaluesensor'
sys_test_context_task = SystemTestContextBuilder().build()
PK_ATTRIBUTE_NAME = 'PK'
SK_ATTRIBUTE_NAME = 'SK'
TABLE_ATTRIBUTES = [{'AttributeName': PK_ATTRIBUTE_NAME, 'AttributeType': 'S'}, {'AttributeName': SK_ATTRIBUTE_NAME, 'AttributeType': 'S'}]
TABLE_KEY_SCHEMA = [{'AttributeName': 'PK', 'KeyType': 'HASH'}, {'AttributeName': 'SK', 'KeyType': 'RANGE'}]
TABLE_THROUGHPUT = {'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}

@task
def create_table(table_name: str):
    if False:
        print('Hello World!')
    ddb = boto3.resource('dynamodb')
    table = ddb.create_table(AttributeDefinitions=TABLE_ATTRIBUTES, TableName=table_name, KeySchema=TABLE_KEY_SCHEMA, ProvisionedThroughput=TABLE_THROUGHPUT)
    boto3.client('dynamodb').get_waiter('table_exists').wait(TableName=table_name)
    table.put_item(Item={'PK': 'Test', 'SK': '2022-07-12T11:11:25-0400', 'Value': 'Testing'})

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_table(table_name: str):
    if False:
        for i in range(10):
            print('nop')
    client = boto3.client('dynamodb')
    client.delete_table(TableName=table_name)
    client.get_waiter('table_not_exists').wait(TableName=table_name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example']) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    table_name = f'{env_id}-dynamodb-table'
    create_table = create_table(table_name=table_name)
    delete_table = delete_table(table_name)
    dynamodb_sensor = DynamoDBValueSensor(task_id='waiting_for_dynamodb_item_value', table_name=table_name, partition_key_name=PK_ATTRIBUTE_NAME, partition_key_value='Test', sort_key_name=SK_ATTRIBUTE_NAME, sort_key_value='2022-07-12T11:11:25-0400', attribute_name='Value', attribute_value='Testing')
    chain(test_context, create_table, dynamodb_sensor, delete_table)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)