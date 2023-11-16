"""
   This DAG will not work unless you create an Amazon EMR cluster running
   Apache Hive and copy data into it following steps 1-4 (inclusive) here:
   https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/EMRforDynamoDB.Tutorial.html
"""
from __future__ import annotations
from datetime import datetime
from airflow.decorators import task
from airflow.models import Connection
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.dynamodb import DynamoDBHook
from airflow.providers.amazon.aws.transfers.hive_to_dynamodb import HiveToDynamoDBOperator
from airflow.utils import db
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
DAG_ID = 'example_hive_to_dynamodb'
HIVE_CONNECTION_ID_KEY = 'HIVE_CONNECTION_ID'
HIVE_HOSTNAME_KEY = 'HIVE_HOSTNAME'
sys_test_context_task = SystemTestContextBuilder().add_variable(HIVE_CONNECTION_ID_KEY).add_variable(HIVE_HOSTNAME_KEY).build()
DYNAMODB_TABLE_HASH_KEY = 'feature_id'
HIVE_SQL = 'SELECT feature_id, feature_name, feature_class, state_alpha FROM hive_features'

@task
def create_dynamodb_table(table_name):
    if False:
        print('Hello World!')
    client = DynamoDBHook(client_type='dynamodb').conn
    client.create_table(TableName=table_name, KeySchema=[{'AttributeName': DYNAMODB_TABLE_HASH_KEY, 'KeyType': 'HASH'}], AttributeDefinitions=[{'AttributeName': DYNAMODB_TABLE_HASH_KEY, 'AttributeType': 'N'}], ProvisionedThroughput={'ReadCapacityUnits': 20, 'WriteCapacityUnits': 20})
    waiter = client.get_waiter('table_exists')
    waiter.wait(TableName=table_name, WaiterConfig={'Delay': 1})

@task
def get_dynamodb_item_count(table_name):
    if False:
        print('Hello World!')
    '\n    A DynamoDB table has an ItemCount value, but it is only updated every six hours.\n    To verify this DAG worked, we will scan the table and count the items manually.\n    '
    table = DynamoDBHook(resource_type='dynamodb').conn.Table(table_name)
    response = table.scan(Select='COUNT')
    item_count = response['Count']
    while 'LastEvaluatedKey' in response:
        response = table.scan(Select='COUNT', ExclusiveStartKey=response['LastEvaluatedKey'])
        item_count += response['Count']
    print(f'DynamoDB table contains {item_count} items.')

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_dynamodb_table(table_name):
    if False:
        for i in range(10):
            print('nop')
    DynamoDBHook(client_type='dynamodb').conn.delete_table(TableName=table_name)

@task
def configure_hive_connection(connection_id, hostname):
    if False:
        while True:
            i = 10
    db.merge_conn(Connection(conn_id=connection_id, conn_type='hiveserver2', host=hostname, port=10000))
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    dynamodb_table_name = f'{env_id}-hive_to_dynamo'
    hive_connection_id = test_context[HIVE_CONNECTION_ID_KEY]
    hive_hostname = test_context[HIVE_HOSTNAME_KEY]
    backup_to_dynamodb = HiveToDynamoDBOperator(task_id='backup_to_dynamodb', hiveserver2_conn_id=hive_connection_id, sql=HIVE_SQL, table_name=dynamodb_table_name, table_keys=[DYNAMODB_TABLE_HASH_KEY])
    chain(test_context, configure_hive_connection(hive_connection_id, hive_hostname), create_dynamodb_table(dynamodb_table_name), backup_to_dynamodb, get_dynamodb_item_count(dynamodb_table_name), delete_dynamodb_table(dynamodb_table_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)