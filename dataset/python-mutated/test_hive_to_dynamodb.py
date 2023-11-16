from __future__ import annotations
import datetime
import json
from unittest import mock
import pandas as pd
from moto import mock_dynamodb
import airflow.providers.amazon.aws.transfers.hive_to_dynamodb
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.dynamodb import DynamoDBHook
DEFAULT_DATE = datetime.datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
DEFAULT_DATE_DS = DEFAULT_DATE_ISO[:10]

class TestHiveToDynamoDBOperator:

    def setup_method(self):
        if False:
            return 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        dag = DAG('test_dag_id', default_args=args)
        self.dag = dag
        self.sql = 'SELECT 1'
        self.hook = DynamoDBHook(aws_conn_id='aws_default', region_name='us-east-1')

    @staticmethod
    def process_data(data, *args, **kwargs):
        if False:
            while True:
                i = 10
        return json.loads(data.to_json(orient='records'))

    @mock_dynamodb
    def test_get_conn_returns_a_boto3_connection(self):
        if False:
            while True:
                i = 10
        hook = DynamoDBHook(aws_conn_id='aws_default')
        assert hook.get_conn() is not None

    @mock.patch('airflow.providers.apache.hive.hooks.hive.HiveServer2Hook.get_pandas_df', return_value=pd.DataFrame(data=[('1', 'sid')], columns=['id', 'name']))
    @mock_dynamodb
    def test_get_records_with_schema(self, mock_get_pandas_df):
        if False:
            while True:
                i = 10
        self.hook.get_conn().create_table(TableName='test_airflow', KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}], AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}], ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10})
        operator = airflow.providers.amazon.aws.transfers.hive_to_dynamodb.HiveToDynamoDBOperator(sql=self.sql, table_name='test_airflow', task_id='hive_to_dynamodb_check', table_keys=['id'], dag=self.dag)
        operator.execute(None)
        table = self.hook.get_conn().Table('test_airflow')
        table.meta.client.get_waiter('table_exists').wait(TableName='test_airflow')
        assert table.item_count == 1

    @mock.patch('airflow.providers.apache.hive.hooks.hive.HiveServer2Hook.get_pandas_df', return_value=pd.DataFrame(data=[('1', 'sid'), ('1', 'gupta')], columns=['id', 'name']))
    @mock_dynamodb
    def test_pre_process_records_with_schema(self, mock_get_pandas_df):
        if False:
            i = 10
            return i + 15
        self.hook.get_conn().create_table(TableName='test_airflow', KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}], AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}], ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10})
        operator = airflow.providers.amazon.aws.transfers.hive_to_dynamodb.HiveToDynamoDBOperator(sql=self.sql, table_name='test_airflow', task_id='hive_to_dynamodb_check', table_keys=['id'], pre_process=self.process_data, dag=self.dag)
        operator.execute(None)
        table = self.hook.get_conn().Table('test_airflow')
        table.meta.client.get_waiter('table_exists').wait(TableName='test_airflow')
        assert table.item_count == 1