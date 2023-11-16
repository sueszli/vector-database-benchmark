from __future__ import annotations
from unittest import mock
import pytest
from moto import mock_glue
from airflow.exceptions import AirflowException, AirflowSkipException, TaskDeferred
from airflow.providers.amazon.aws.hooks.glue_catalog import GlueCatalogHook
from airflow.providers.amazon.aws.sensors.glue_catalog_partition import GlueCatalogPartitionSensor

class TestGlueCatalogPartitionSensor:
    task_id = 'test_glue_catalog_partition_sensor'

    @mock_glue
    @mock.patch.object(GlueCatalogHook, 'check_for_partition')
    def test_poke(self, mock_check_for_partition):
        if False:
            print('Hello World!')
        mock_check_for_partition.return_value = True
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl')
        assert op.poke({})

    @mock_glue
    @mock.patch.object(GlueCatalogHook, 'check_for_partition')
    def test_poke_false(self, mock_check_for_partition):
        if False:
            for i in range(10):
                print('nop')
        mock_check_for_partition.return_value = False
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl')
        assert not op.poke({})

    @mock_glue
    @mock.patch.object(GlueCatalogHook, 'check_for_partition')
    def test_poke_default_args(self, mock_check_for_partition):
        if False:
            while True:
                i = 10
        table_name = 'test_glue_catalog_partition_sensor_tbl'
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name=table_name)
        op.poke({})
        assert op.hook.region_name is None
        assert op.hook.aws_conn_id == 'aws_default'
        mock_check_for_partition.assert_called_once_with('default', table_name, "ds='{{ ds }}'")

    @mock_glue
    @mock.patch.object(GlueCatalogHook, 'check_for_partition')
    def test_poke_nondefault_args(self, mock_check_for_partition):
        if False:
            return 10
        table_name = 'my_table'
        expression = 'col=val'
        aws_conn_id = 'my_aws_conn_id'
        region_name = 'us-west-2'
        database_name = 'my_db'
        poke_interval = 2
        timeout = 3
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name=table_name, expression=expression, aws_conn_id=aws_conn_id, region_name=region_name, database_name=database_name, poke_interval=poke_interval, timeout=timeout)
        op.hook.get_connection = lambda _: None
        op.poke({})
        assert op.hook.region_name == region_name
        assert op.hook.aws_conn_id == aws_conn_id
        assert op.poke_interval == poke_interval
        assert op.timeout == timeout
        mock_check_for_partition.assert_called_once_with(database_name, table_name, expression)

    @mock_glue
    @mock.patch.object(GlueCatalogHook, 'check_for_partition')
    def test_dot_notation(self, mock_check_for_partition):
        if False:
            i = 10
            return i + 15
        db_table = 'my_db.my_tbl'
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name=db_table)
        op.poke({})
        mock_check_for_partition.assert_called_once_with('my_db', 'my_tbl', "ds='{{ ds }}'")

    def test_deferrable_mode_raises_task_deferred(self):
        if False:
            print('Hello World!')
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl', deferrable=True)
        with pytest.raises(TaskDeferred):
            op.execute({})

    def test_execute_complete_fails_if_status_is_not_success(self):
        if False:
            for i in range(10):
                print('nop')
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl', deferrable=True)
        event = {'status': 'FAILED'}
        with pytest.raises(AirflowException):
            op.execute_complete(context={}, event=event)

    def test_execute_complete_succeeds_if_status_is_success(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl', deferrable=True)
        event = {'status': 'success'}
        op.execute_complete(context={}, event=event)
        assert 'Partition exists in the Glue Catalog' in caplog.messages

    @pytest.mark.parametrize('soft_fail, expected_exception', ((False, AirflowException), (True, AirflowSkipException)))
    def test_fail_execute_complete(self, soft_fail, expected_exception):
        if False:
            return 10
        op = GlueCatalogPartitionSensor(task_id=self.task_id, table_name='tbl', deferrable=True)
        op.soft_fail = soft_fail
        event = {'status': 'Failed'}
        message = f'Trigger error: event is {event}'
        with pytest.raises(expected_exception, match=message):
            op.execute_complete(context={}, event=event)