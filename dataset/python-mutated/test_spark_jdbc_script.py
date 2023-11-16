from __future__ import annotations
from unittest import mock
import pytest
from pyspark.sql.readwriter import DataFrameReader, DataFrameWriter
from airflow.providers.apache.spark.hooks.spark_jdbc_script import SPARK_READ_FROM_JDBC, SPARK_WRITE_TO_JDBC, _create_spark_session, _parse_arguments, _run_spark, spark_read_from_jdbc, spark_write_to_jdbc

@pytest.fixture()
def mock_spark_session():
    if False:
        return 10
    with mock.patch('airflow.providers.apache.spark.hooks.spark_jdbc_script.SparkSession') as mok:
        yield mok

class TestSparkJDBCScrip:
    jdbc_arguments = ['-cmdType', 'spark_to_jdbc', '-url', 'jdbc:postgresql://localhost:5432/default', '-user', 'user', '-password', 'supersecret', '-metastoreTable', 'hiveMcHiveFace', '-jdbcTable', 'tableMcTableFace', '-jdbcDriver', 'org.postgresql.Driver', '-jdbcTruncate', 'false', '-saveMode', 'append', '-saveFormat', 'parquet', '-batchsize', '100', '-fetchsize', '200', '-name', 'airflow-spark-jdbc-script-test', '-numPartitions', '10', '-partitionColumn', 'columnMcColumnFace', '-lowerBound', '10', '-upperBound', '20', '-createTableColumnTypes', 'columnMcColumnFace INTEGER(100), name CHAR(64),comments VARCHAR(1024)']
    default_arguments = {'cmd_type': 'spark_to_jdbc', 'url': 'jdbc:postgresql://localhost:5432/default', 'user': 'user', 'password': 'supersecret', 'metastore_table': 'hiveMcHiveFace', 'jdbc_table': 'tableMcTableFace', 'jdbc_driver': 'org.postgresql.Driver', 'truncate': 'false', 'save_mode': 'append', 'save_format': 'parquet', 'batch_size': '100', 'fetch_size': '200', 'name': 'airflow-spark-jdbc-script-test', 'num_partitions': '10', 'partition_column': 'columnMcColumnFace', 'lower_bound': '10', 'upper_bound': '20', 'create_table_column_types': 'columnMcColumnFace INTEGER(100), name CHAR(64),comments VARCHAR(1024)'}

    def test_parse_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        parsed_arguments = _parse_arguments(args=self.jdbc_arguments)
        for (argument_name, argument_value) in self.default_arguments.items():
            assert getattr(parsed_arguments, argument_name) == argument_value

    @mock.patch('airflow.providers.apache.spark.hooks.spark_jdbc_script.spark_write_to_jdbc')
    def test_run_spark_write_to_jdbc(self, mock_spark_write_to_jdbc, mock_spark_session):
        if False:
            print('Hello World!')
        arguments = _parse_arguments(['-cmdType', SPARK_WRITE_TO_JDBC] + self.jdbc_arguments[2:])
        spark_session = mock_spark_session.builder.appName(arguments.name).enableHiveSupport().getOrCreate()
        _run_spark(arguments=arguments)
        mock_spark_write_to_jdbc.assert_called_once_with(spark_session, arguments.url, arguments.user, arguments.password, arguments.metastore_table, arguments.jdbc_table, arguments.jdbc_driver, arguments.truncate, arguments.save_mode, arguments.batch_size, arguments.num_partitions, arguments.create_table_column_types)

    @mock.patch('airflow.providers.apache.spark.hooks.spark_jdbc_script.spark_read_from_jdbc')
    def test_run_spark_read_from_jdbc(self, mock_spark_read_from_jdbc, mock_spark_session):
        if False:
            while True:
                i = 10
        arguments = _parse_arguments(['-cmdType', SPARK_READ_FROM_JDBC] + self.jdbc_arguments[2:])
        spark_session = mock_spark_session.builder.appName(arguments.name).enableHiveSupport().getOrCreate()
        _run_spark(arguments=arguments)
        mock_spark_read_from_jdbc.assert_called_once_with(spark_session, arguments.url, arguments.user, arguments.password, arguments.metastore_table, arguments.jdbc_table, arguments.jdbc_driver, arguments.save_mode, arguments.save_format, arguments.fetch_size, arguments.num_partitions, arguments.partition_column, arguments.lower_bound, arguments.upper_bound)

    @pytest.mark.system('spark')
    @mock.patch.object(DataFrameWriter, 'save')
    def test_spark_write_to_jdbc(self, mock_writer_save):
        if False:
            for i in range(10):
                print('nop')
        arguments = _parse_arguments(self.jdbc_arguments)
        spark_session = _create_spark_session(arguments)
        spark_session.sql(f'CREATE TABLE IF NOT EXISTS {arguments.metastore_table} (key INT)')
        spark_write_to_jdbc(spark_session=spark_session, url=arguments.url, user=arguments.user, password=arguments.password, metastore_table=arguments.metastore_table, jdbc_table=arguments.jdbc_table, driver=arguments.jdbc_driver, truncate=arguments.truncate, save_mode=arguments.save_mode, batch_size=arguments.batch_size, num_partitions=arguments.num_partitions, create_table_column_types=arguments.create_table_column_types)
        mock_writer_save.assert_called_once_with(mode=arguments.save_mode)

    @pytest.mark.system('spark')
    @mock.patch.object(DataFrameReader, 'load')
    def test_spark_read_from_jdbc(self, mock_reader_load):
        if False:
            return 10
        arguments = _parse_arguments(self.jdbc_arguments)
        spark_session = _create_spark_session(arguments)
        spark_session.sql(f'CREATE TABLE IF NOT EXISTS {arguments.metastore_table} (key INT)')
        spark_read_from_jdbc(spark_session, arguments.url, arguments.user, arguments.password, arguments.metastore_table, arguments.jdbc_table, arguments.jdbc_driver, arguments.save_mode, arguments.save_format, arguments.fetch_size, arguments.num_partitions, arguments.partition_column, arguments.lower_bound, arguments.upper_bound)
        mock_reader_load().write.saveAsTable.assert_called_once_with(arguments.metastore_table, format=arguments.save_format, mode=arguments.save_mode)