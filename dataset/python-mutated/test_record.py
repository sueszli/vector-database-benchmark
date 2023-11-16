from __future__ import annotations
from unittest.mock import patch
from airflow.providers.apache.cassandra.sensors.record import CassandraRecordSensor
TEST_CASSANDRA_CONN_ID = 'cassandra_default'
TEST_CASSANDRA_TABLE = 't'
TEST_CASSANDRA_KEY = {'foo': 'bar'}

class TestCassandraRecordSensor:

    @patch('airflow.providers.apache.cassandra.sensors.record.CassandraHook')
    def test_poke(self, mock_hook):
        if False:
            return 10
        sensor = CassandraRecordSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE, keys=TEST_CASSANDRA_KEY)
        exists = sensor.poke(dict())
        assert exists
        mock_hook.return_value.record_exists.assert_called_once_with(TEST_CASSANDRA_TABLE, TEST_CASSANDRA_KEY)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.record.CassandraHook')
    def test_poke_should_not_fail_with_empty_keys(self, mock_hook):
        if False:
            i = 10
            return i + 15
        sensor = CassandraRecordSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE, keys=None)
        exists = sensor.poke(dict())
        assert exists
        mock_hook.return_value.record_exists.assert_called_once_with(TEST_CASSANDRA_TABLE, None)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.record.CassandraHook')
    def test_poke_should_return_false_for_non_existing_table(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_hook.return_value.record_exists.return_value = False
        sensor = CassandraRecordSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE, keys=TEST_CASSANDRA_KEY)
        exists = sensor.poke(dict())
        assert not exists
        mock_hook.return_value.record_exists.assert_called_once_with(TEST_CASSANDRA_TABLE, TEST_CASSANDRA_KEY)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.record.CassandraHook')
    def test_init_with_default_conn(self, mock_hook):
        if False:
            return 10
        sensor = CassandraRecordSensor(task_id='test_task', table=TEST_CASSANDRA_TABLE, keys=TEST_CASSANDRA_KEY)
        assert sensor.cassandra_conn_id == TEST_CASSANDRA_CONN_ID