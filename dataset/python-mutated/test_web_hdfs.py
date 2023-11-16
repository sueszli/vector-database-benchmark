from __future__ import annotations
from unittest import mock
from airflow.providers.apache.hdfs.sensors.web_hdfs import WebHdfsSensor
TEST_HDFS_CONN = 'webhdfs_default'
TEST_HDFS_PATH = 'hdfs://user/hive/warehouse/airflow.db/static_babynames'

class TestWebHdfsSensor:

    @mock.patch('airflow.providers.apache.hdfs.hooks.webhdfs.WebHDFSHook')
    def test_poke(self, mock_hook):
        if False:
            while True:
                i = 10
        sensor = WebHdfsSensor(task_id='test_task', webhdfs_conn_id=TEST_HDFS_CONN, filepath=TEST_HDFS_PATH)
        exists = sensor.poke(dict())
        assert exists
        mock_hook.return_value.check_for_path.assert_called_once_with(hdfs_path=TEST_HDFS_PATH)
        mock_hook.assert_called_once_with(TEST_HDFS_CONN)

    @mock.patch('airflow.providers.apache.hdfs.hooks.webhdfs.WebHDFSHook')
    def test_poke_should_return_false_for_non_existing_table(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        mock_hook.return_value.check_for_path.return_value = False
        sensor = WebHdfsSensor(task_id='test_task', webhdfs_conn_id=TEST_HDFS_CONN, filepath=TEST_HDFS_PATH)
        exists = sensor.poke(dict())
        assert not exists
        mock_hook.return_value.check_for_path.assert_called_once_with(hdfs_path=TEST_HDFS_PATH)
        mock_hook.assert_called_once_with(TEST_HDFS_CONN)