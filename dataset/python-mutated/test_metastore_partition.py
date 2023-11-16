from __future__ import annotations
import os
from unittest import mock
import pytest
from airflow.providers.apache.hive.sensors.metastore_partition import MetastorePartitionSensor
from tests.providers.apache.hive import DEFAULT_DATE, DEFAULT_DATE_DS, MockDBConnection, TestHiveEnvironment

@pytest.mark.skipif('AIRFLOW_RUNALL_TESTS' not in os.environ, reason='Skipped because AIRFLOW_RUNALL_TESTS is not set')
class TestHivePartitionSensor(TestHiveEnvironment):

    def test_hive_metastore_sql_sensor(self):
        if False:
            while True:
                i = 10
        op = MetastorePartitionSensor(task_id='hive_partition_check', conn_id='test_connection_id', sql='test_sql', table='airflow.static_babynames_partitioned', partition_name=f'ds={DEFAULT_DATE_DS}', dag=self.dag)
        op._get_hook = mock.MagicMock(return_value=MockDBConnection({}))
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)