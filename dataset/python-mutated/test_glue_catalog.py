from __future__ import annotations
from unittest import mock
import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_glue
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.glue_catalog import GlueCatalogHook
DB_NAME = 'db'
TABLE_NAME = 'table'
TABLE_INPUT = {'Name': TABLE_NAME, 'StorageDescriptor': {'Columns': [{'Name': 'string', 'Type': 'string', 'Comment': 'string'}], 'Location': f's3://mybucket/{DB_NAME}/{TABLE_NAME}'}}
PARTITION_INPUT: dict = {'Values': []}

@mock_glue
class TestGlueCatalogHook:

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.client = boto3.client('glue', region_name='us-east-1')
        self.hook = GlueCatalogHook(region_name='us-east-1')

    def test_get_conn_returns_a_boto3_connection(self):
        if False:
            return 10
        hook = GlueCatalogHook(region_name='us-east-1')
        assert hook.get_conn() is not None

    def test_conn_id(self):
        if False:
            print('Hello World!')
        hook = GlueCatalogHook(aws_conn_id='my_aws_conn_id', region_name='us-east-1')
        assert hook.aws_conn_id == 'my_aws_conn_id'

    def test_region(self):
        if False:
            while True:
                i = 10
        hook = GlueCatalogHook(region_name='us-west-2')
        assert hook.region_name == 'us-west-2'

    @mock.patch.object(GlueCatalogHook, 'get_conn')
    def test_get_partitions_empty(self, mock_get_conn):
        if False:
            print('Hello World!')
        response = set()
        mock_get_conn.get_paginator.paginate.return_value = response
        hook = GlueCatalogHook(region_name='us-east-1')
        assert hook.get_partitions('db', 'tbl') == set()

    @mock.patch.object(GlueCatalogHook, 'get_conn')
    def test_get_partitions(self, mock_get_conn):
        if False:
            while True:
                i = 10
        response = [{'Partitions': [{'Values': ['2015-01-01']}]}]
        mock_paginator = mock.Mock()
        mock_paginator.paginate.return_value = response
        mock_conn = mock.Mock()
        mock_conn.get_paginator.return_value = mock_paginator
        mock_get_conn.return_value = mock_conn
        hook = GlueCatalogHook(region_name='us-east-1')
        result = hook.get_partitions('db', 'tbl', expression='foo=bar', page_size=2, max_items=3)
        assert result == {('2015-01-01',)}
        mock_conn.get_paginator.assert_called_once_with('get_partitions')
        mock_paginator.paginate.assert_called_once_with(DatabaseName='db', TableName='tbl', Expression='foo=bar', PaginationConfig={'PageSize': 2, 'MaxItems': 3})

    @mock.patch.object(GlueCatalogHook, 'get_partitions')
    def test_check_for_partition(self, mock_get_partitions):
        if False:
            i = 10
            return i + 15
        mock_get_partitions.return_value = {('2018-01-01',)}
        hook = GlueCatalogHook(region_name='us-east-1')
        assert hook.check_for_partition('db', 'tbl', 'expr')
        mock_get_partitions.assert_called_once_with('db', 'tbl', 'expr', max_items=1)

    @mock.patch.object(GlueCatalogHook, 'get_partitions')
    def test_check_for_partition_false(self, mock_get_partitions):
        if False:
            return 10
        mock_get_partitions.return_value = set()
        hook = GlueCatalogHook(region_name='us-east-1')
        assert not hook.check_for_partition('db', 'tbl', 'expr')

    def test_get_table_exists(self):
        if False:
            while True:
                i = 10
        self.client.create_database(DatabaseInput={'Name': DB_NAME})
        self.client.create_table(DatabaseName=DB_NAME, TableInput=TABLE_INPUT)
        result = self.hook.get_table(DB_NAME, TABLE_NAME)
        assert result['Name'] == TABLE_INPUT['Name']
        assert result['StorageDescriptor']['Location'] == TABLE_INPUT['StorageDescriptor']['Location']

    def test_get_table_not_exists(self):
        if False:
            return 10
        self.client.create_database(DatabaseInput={'Name': DB_NAME})
        self.client.create_table(DatabaseName=DB_NAME, TableInput=TABLE_INPUT)
        with pytest.raises(Exception):
            self.hook.get_table(DB_NAME, 'dummy_table')

    def test_get_table_location(self):
        if False:
            return 10
        self.client.create_database(DatabaseInput={'Name': DB_NAME})
        self.client.create_table(DatabaseName=DB_NAME, TableInput=TABLE_INPUT)
        result = self.hook.get_table_location(DB_NAME, TABLE_NAME)
        assert result == TABLE_INPUT['StorageDescriptor']['Location']

    def test_get_partition(self):
        if False:
            print('Hello World!')
        self.client.create_database(DatabaseInput={'Name': DB_NAME})
        self.client.create_table(DatabaseName=DB_NAME, TableInput=TABLE_INPUT)
        self.client.create_partition(DatabaseName=DB_NAME, TableName=TABLE_NAME, PartitionInput=PARTITION_INPUT)
        result = self.hook.get_partition(DB_NAME, TABLE_NAME, PARTITION_INPUT['Values'])
        assert result['Values'] == PARTITION_INPUT['Values']
        assert result['DatabaseName'] == DB_NAME
        assert result['TableName'] == TABLE_INPUT['Name']

    @mock.patch.object(GlueCatalogHook, 'get_conn')
    def test_get_partition_with_client_error(self, mocked_connection):
        if False:
            while True:
                i = 10
        mocked_client = mock.Mock()
        mocked_client.get_partition.side_effect = ClientError({}, 'get_partition')
        mocked_connection.return_value = mocked_client
        with pytest.raises(AirflowException):
            self.hook.get_partition(DB_NAME, TABLE_NAME, PARTITION_INPUT['Values'])
        mocked_client.get_partition.assert_called_once_with(DatabaseName=DB_NAME, TableName=TABLE_NAME, PartitionValues=PARTITION_INPUT['Values'])

    def test_create_partition(self):
        if False:
            i = 10
            return i + 15
        self.client.create_database(DatabaseInput={'Name': DB_NAME})
        self.client.create_table(DatabaseName=DB_NAME, TableInput=TABLE_INPUT)
        result = self.hook.create_partition(DB_NAME, TABLE_NAME, PARTITION_INPUT)
        assert result

    @mock.patch.object(GlueCatalogHook, 'get_conn')
    def test_create_partition_with_client_error(self, mocked_connection):
        if False:
            while True:
                i = 10
        mocked_client = mock.Mock()
        mocked_client.create_partition.side_effect = ClientError({}, 'create_partition')
        mocked_connection.return_value = mocked_client
        with pytest.raises(AirflowException):
            self.hook.create_partition(DB_NAME, TABLE_NAME, PARTITION_INPUT)
        mocked_client.create_partition.assert_called_once_with(DatabaseName=DB_NAME, TableName=TABLE_NAME, PartitionInput=PARTITION_INPUT)