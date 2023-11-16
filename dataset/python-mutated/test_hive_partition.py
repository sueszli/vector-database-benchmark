from __future__ import annotations
import os
from unittest.mock import patch
import pytest
from airflow.providers.apache.hive.sensors.hive_partition import HivePartitionSensor
from tests.providers.apache.hive import DEFAULT_DATE, MockHiveMetastoreHook, TestHiveEnvironment

@pytest.mark.skipif('AIRFLOW_RUNALL_TESTS' not in os.environ, reason='Skipped because AIRFLOW_RUNALL_TESTS is not set')
@patch('airflow.providers.apache.hive.sensors.hive_partition.HiveMetastoreHook', side_effect=MockHiveMetastoreHook)
class TestHivePartitionSensor(TestHiveEnvironment):

    def test_hive_partition_sensor(self, mock_hive_metastore_hook):
        if False:
            print('Hello World!')
        op = HivePartitionSensor(task_id='hive_partition_check', table='airflow.static_babynames_partitioned', dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)