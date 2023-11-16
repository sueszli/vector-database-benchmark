from __future__ import annotations
from unittest.mock import patch
from airflow.providers.apache.cassandra.sensors.table import CassandraTableSensor
TEST_CASSANDRA_CONN_ID = 'cassandra_default'
TEST_CASSANDRA_TABLE = 't'
TEST_CASSANDRA_TABLE_WITH_KEYSPACE = 'keyspacename.tablename'

class TestCassandraTableSensor:

    @patch('airflow.providers.apache.cassandra.sensors.table.CassandraHook')
    def test_poke(self, mock_hook):
        if False:
            while True:
                i = 10
        sensor = CassandraTableSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE)
        exists = sensor.poke(dict())
        assert exists
        mock_hook.return_value.table_exists.assert_called_once_with(TEST_CASSANDRA_TABLE)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.table.CassandraHook')
    def test_poke_should_return_false_for_non_existing_table(self, mock_hook):
        if False:
            return 10
        mock_hook.return_value.table_exists.return_value = False
        sensor = CassandraTableSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE)
        exists = sensor.poke(dict())
        assert not exists
        mock_hook.return_value.table_exists.assert_called_once_with(TEST_CASSANDRA_TABLE)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.table.CassandraHook')
    def test_poke_should_succeed_for_table_with_mentioned_keyspace(self, mock_hook):
        if False:
            i = 10
            return i + 15
        sensor = CassandraTableSensor(task_id='test_task', cassandra_conn_id=TEST_CASSANDRA_CONN_ID, table=TEST_CASSANDRA_TABLE_WITH_KEYSPACE)
        exists = sensor.poke(dict())
        assert exists
        mock_hook.return_value.table_exists.assert_called_once_with(TEST_CASSANDRA_TABLE_WITH_KEYSPACE)
        mock_hook.assert_called_once_with(TEST_CASSANDRA_CONN_ID)

    @patch('airflow.providers.apache.cassandra.sensors.table.CassandraHook')
    def test_init_with_default_conn(self, mock_hook):
        if False:
            while True:
                i = 10
        sensor = CassandraTableSensor(task_id='test_task', table=TEST_CASSANDRA_TABLE)
        assert sensor.cassandra_conn_id == TEST_CASSANDRA_CONN_ID