from __future__ import annotations
from unittest.mock import patch
import pytest
from airflow.models import Connection
from airflow.providers.apache.spark.hooks.spark_jdbc import SparkJDBCHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestSparkJDBCHook:
    _config = {'cmd_type': 'spark_to_jdbc', 'jdbc_table': 'tableMcTableFace', 'jdbc_driver': 'org.postgresql.Driver', 'metastore_table': 'hiveMcHiveFace', 'jdbc_truncate': False, 'save_mode': 'append', 'save_format': 'parquet', 'batch_size': 100, 'fetch_size': 200, 'num_partitions': 10, 'partition_column': 'columnMcColumnFace', 'lower_bound': '10', 'upper_bound': '20', 'create_table_column_types': 'columnMcColumnFace INTEGER(100), name CHAR(64),comments VARCHAR(1024)'}
    _invalid_config = {'cmd_type': 'spark_to_jdbc', 'jdbc_table': 'tableMcTableFace', 'jdbc_driver': 'org.postgresql.Driver', 'metastore_table': 'hiveMcHiveFace', 'jdbc_truncate': False, 'save_mode': 'append', 'save_format': 'parquet', 'batch_size': 100, 'fetch_size': 200, 'num_partitions': 10, 'partition_column': 'columnMcColumnFace', 'upper_bound': '20', 'create_table_column_types': 'columnMcColumnFace INTEGER(100), name CHAR(64),comments VARCHAR(1024)'}

    def setup_method(self):
        if False:
            print('Hello World!')
        db.merge_conn(Connection(conn_id='spark-default', conn_type='spark', host='yarn://yarn-master', extra='{"queue": "root.etl", "deploy-mode": "cluster"}'))
        db.merge_conn(Connection(conn_id='jdbc-default', conn_type='postgres', host='localhost', schema='default', port=5432, login='user', password='supersecret', extra='{"conn_prefix":"jdbc:postgresql://"}'))
        db.merge_conn(Connection(conn_id='jdbc-invalid-host', conn_type='postgres', host='localhost/test', schema='default', port=5432, login='user', password='supersecret', extra='{"conn_prefix":"jdbc:postgresql://"}'))
        db.merge_conn(Connection(conn_id='jdbc-invalid-schema', conn_type='postgres', host='localhost', schema='default?test=', port=5432, login='user', password='supersecret', extra='{"conn_prefix":"jdbc:postgresql://"}'))
        db.merge_conn(Connection(conn_id='jdbc-invalid-extra-conn-prefix', conn_type='postgres', host='localhost', schema='default', port=5432, login='user', password='supersecret', extra='{"conn_prefix":"jdbc:mysql://some_host:8085/test?some_query_param=true#"}'))

    def test_resolve_jdbc_connection(self):
        if False:
            while True:
                i = 10
        hook = SparkJDBCHook(jdbc_conn_id='jdbc-default')
        expected_connection = {'url': 'localhost:5432', 'schema': 'default', 'conn_prefix': 'jdbc:postgresql://', 'user': 'user', 'password': 'supersecret'}
        connection = hook._resolve_jdbc_connection()
        assert connection == expected_connection

    def test_build_jdbc_arguments(self):
        if False:
            while True:
                i = 10
        hook = SparkJDBCHook(**self._config)
        cmd = hook._build_jdbc_application_arguments(hook._resolve_jdbc_connection())
        expected_jdbc_arguments = ['-cmdType', 'spark_to_jdbc', '-url', 'jdbc:postgresql://localhost:5432/default', '-user', 'user', '-password', 'supersecret', '-metastoreTable', 'hiveMcHiveFace', '-jdbcTable', 'tableMcTableFace', '-jdbcDriver', 'org.postgresql.Driver', '-batchsize', '100', '-fetchsize', '200', '-numPartitions', '10', '-partitionColumn', 'columnMcColumnFace', '-lowerBound', '10', '-upperBound', '20', '-saveMode', 'append', '-saveFormat', 'parquet', '-createTableColumnTypes', 'columnMcColumnFace INTEGER(100), name CHAR(64),comments VARCHAR(1024)']
        assert expected_jdbc_arguments == cmd

    def test_build_jdbc_arguments_invalid(self):
        if False:
            while True:
                i = 10
        hook = SparkJDBCHook(**self._invalid_config)
        hook._build_jdbc_application_arguments(hook._resolve_jdbc_connection())

    def test_invalid_host(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='host should not contain a'):
            SparkJDBCHook(jdbc_conn_id='jdbc-invalid-host', **self._config)

    def test_invalid_schema(self):
        if False:
            return 10
        with pytest.raises(ValueError, match='schema should not contain a'):
            SparkJDBCHook(jdbc_conn_id='jdbc-invalid-schema', **self._config)

    @patch('airflow.providers.apache.spark.hooks.spark_submit.SparkSubmitHook.submit')
    def test_invalid_extra_conn_prefix(self, mock_submit):
        if False:
            return 10
        hook = SparkJDBCHook(jdbc_conn_id='jdbc-invalid-extra-conn-prefix', **self._config)
        with pytest.raises(ValueError, match='extra conn_prefix should not contain a'):
            hook.submit_jdbc_job()