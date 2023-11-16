from __future__ import annotations
import datetime
from unittest import mock
from airflow.models.dag import DAG
from airflow.providers.microsoft.azure.operators.wasb_delete_blob import WasbDeleteBlobOperator

class TestWasbDeleteBlobOperator:
    _config = {'container_name': 'container', 'blob_name': 'blob'}

    def setup_method(self):
        if False:
            while True:
                i = 10
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        operator = WasbDeleteBlobOperator(task_id='wasb_operator_1', dag=self.dag, **self._config)
        assert operator.container_name == self._config['container_name']
        assert operator.blob_name == self._config['blob_name']
        assert operator.is_prefix is False
        assert operator.ignore_if_missing is False
        operator = WasbDeleteBlobOperator(task_id='wasb_operator_2', dag=self.dag, is_prefix=True, ignore_if_missing=True, **self._config)
        assert operator.is_prefix is True
        assert operator.ignore_if_missing is True

    @mock.patch('airflow.providers.microsoft.azure.operators.wasb_delete_blob.WasbHook', autospec=True)
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_instance = mock_hook.return_value
        operator = WasbDeleteBlobOperator(task_id='wasb_operator', dag=self.dag, is_prefix=True, ignore_if_missing=True, **self._config)
        operator.execute(None)
        mock_instance.delete_file.assert_called_once_with('container', 'blob', True, True)