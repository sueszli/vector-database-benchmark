from __future__ import annotations
from unittest import mock
from airflow.providers.influxdb.operators.influxdb import InfluxDBOperator

class TestInfluxDBOperator:

    @mock.patch('airflow.providers.influxdb.operators.influxdb.InfluxDBHook')
    def test_influxdb_operator_test(self, mock_hook):
        if False:
            print('Hello World!')
        sql = 'from(bucket:"test") |> range(start: -10m)'
        op = InfluxDBOperator(task_id='basic_influxdb', sql=sql, influxdb_conn_id='influxdb_default')
        op.execute(mock.MagicMock())
        mock_hook.assert_called_once_with(conn_id='influxdb_default')
        mock_hook.return_value.query.assert_called_once_with(sql)