from __future__ import annotations
import os
from datetime import timedelta
from unittest import mock
import pytest
from airflow.exceptions import AirflowSensorTimeout
from airflow.models.dag import DAG
from airflow.providers.apache.hive.sensors.named_hive_partition import NamedHivePartitionSensor
from airflow.utils.timezone import datetime
from tests.providers.apache.hive import MockHiveMetastoreHook, TestHiveEnvironment
DEFAULT_DATE = datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
DEFAULT_DATE_DS = DEFAULT_DATE_ISO[:10]
pytestmark = pytest.mark.db_test

class TestNamedHivePartitionSensor:

    def setup_method(self):
        if False:
            while True:
                i = 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_dag_id', default_args=args)
        self.next_day = (DEFAULT_DATE + timedelta(days=1)).isoformat()[:10]
        self.database = 'airflow'
        self.partition_by = 'ds'
        self.table = 'static_babynames_partitioned'
        self.hql = "\n                CREATE DATABASE IF NOT EXISTS {{ params.database }};\n                USE {{ params.database }};\n                DROP TABLE IF EXISTS {{ params.table }};\n                CREATE TABLE IF NOT EXISTS {{ params.table }} (\n                    state string,\n                    year string,\n                    name string,\n                    gender string,\n                    num int)\n                PARTITIONED BY ({{ params.partition_by }} string);\n                ALTER TABLE {{ params.table }}\n                ADD PARTITION({{ params.partition_by }}='{{ ds }}');\n                "
        self.hook = MockHiveMetastoreHook()

    def test_parse_partition_name_correct(self):
        if False:
            while True:
                i = 10
        schema = 'default'
        table = 'users'
        partition = 'ds=2016-01-01/state=IT'
        name = f'{schema}.{table}/{partition}'
        (parsed_schema, parsed_table, parsed_partition) = NamedHivePartitionSensor.parse_partition_name(name)
        assert schema == parsed_schema
        assert table == parsed_table
        assert partition == parsed_partition

    def test_parse_partition_name_incorrect(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'incorrect.name'
        with pytest.raises(ValueError):
            NamedHivePartitionSensor.parse_partition_name(name)

    def test_parse_partition_name_default(self):
        if False:
            i = 10
            return i + 15
        table = 'users'
        partition = 'ds=2016-01-01/state=IT'
        name = f'{table}/{partition}'
        (parsed_schema, parsed_table, parsed_partition) = NamedHivePartitionSensor.parse_partition_name(name)
        assert 'default' == parsed_schema
        assert table == parsed_table
        assert partition == parsed_partition

    def test_poke_existing(self):
        if False:
            i = 10
            return i + 15
        self.hook.metastore.__enter__().check_for_named_partition.return_value = True
        partitions = [f'{self.database}.{self.table}/{self.partition_by}={DEFAULT_DATE_DS}']
        sensor = NamedHivePartitionSensor(partition_names=partitions, task_id='test_poke_existing', poke_interval=1, hook=self.hook, dag=self.dag)
        assert sensor.poke(None)
        self.hook.metastore.__enter__().check_for_named_partition.assert_called_with(self.database, self.table, f'{self.partition_by}={DEFAULT_DATE_DS}')

    def test_poke_non_existing(self):
        if False:
            return 10
        self.hook.metastore.__enter__().check_for_named_partition.return_value = False
        partitions = [f'{self.database}.{self.table}/{self.partition_by}={self.next_day}']
        sensor = NamedHivePartitionSensor(partition_names=partitions, task_id='test_poke_non_existing', poke_interval=1, hook=self.hook, dag=self.dag)
        assert not sensor.poke(None)
        self.hook.metastore.__enter__().check_for_named_partition.assert_called_with(self.database, self.table, f'{self.partition_by}={self.next_day}')

@pytest.mark.skipif('AIRFLOW_RUNALL_TESTS' not in os.environ, reason='Skipped because AIRFLOW_RUNALL_TESTS is not set')
class TestPartitions(TestHiveEnvironment):

    def test_succeeds_on_one_partition(self):
        if False:
            return 10
        mock_hive_metastore_hook = MockHiveMetastoreHook()
        mock_hive_metastore_hook.check_for_named_partition = mock.MagicMock(return_value=True)
        op = NamedHivePartitionSensor(task_id='hive_partition_check', partition_names=['airflow.static_babynames_partitioned/ds={{ds}}'], dag=self.dag, hook=mock_hive_metastore_hook)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)
        mock_hive_metastore_hook.check_for_named_partition.assert_called_once_with('airflow', 'static_babynames_partitioned', 'ds=2015-01-01')

    def test_succeeds_on_multiple_partitions(self):
        if False:
            i = 10
            return i + 15
        mock_hive_metastore_hook = MockHiveMetastoreHook()
        mock_hive_metastore_hook.check_for_named_partition = mock.MagicMock(return_value=True)
        op = NamedHivePartitionSensor(task_id='hive_partition_check', partition_names=['airflow.static_babynames_partitioned/ds={{ds}}', 'airflow.static_babynames_partitioned2/ds={{ds}}'], dag=self.dag, hook=mock_hive_metastore_hook)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)
        mock_hive_metastore_hook.check_for_named_partition.assert_any_call('airflow', 'static_babynames_partitioned', 'ds=2015-01-01')
        mock_hive_metastore_hook.check_for_named_partition.assert_any_call('airflow', 'static_babynames_partitioned2', 'ds=2015-01-01')

    def test_parses_partitions_with_periods(self):
        if False:
            return 10
        name = NamedHivePartitionSensor.parse_partition_name(partition='schema.table/part1=this.can.be.an.issue/part2=ok')
        assert name[0] == 'schema'
        assert name[1] == 'table'
        assert name[2] == 'part1=this.can.be.an.issue/part2=ok'

    def test_times_out_on_nonexistent_partition(self):
        if False:
            while True:
                i = 10
        with pytest.raises(AirflowSensorTimeout):
            mock_hive_metastore_hook = MockHiveMetastoreHook()
            mock_hive_metastore_hook.check_for_named_partition = mock.MagicMock(return_value=False)
            op = NamedHivePartitionSensor(task_id='hive_partition_check', partition_names=['airflow.static_babynames_partitioned/ds={{ds}}', 'airflow.static_babynames_partitioned/ds=nonexistent'], poke_interval=0.1, timeout=1, dag=self.dag, hook=mock_hive_metastore_hook)
            op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)