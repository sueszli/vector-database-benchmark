from __future__ import annotations
from unittest.mock import patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.cloudant.hooks.cloudant import CloudantHook
pytestmark = pytest.mark.db_test

class TestCloudantHook:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.cloudant_hook = CloudantHook()

    @patch('airflow.providers.cloudant.hooks.cloudant.CloudantHook.get_connection', return_value=Connection(login='user', password='password', host='account'))
    @patch('airflow.providers.cloudant.hooks.cloudant.cloudant')
    def test_get_conn(self, mock_cloudant, mock_get_connection):
        if False:
            for i in range(10):
                print('nop')
        cloudant_session = self.cloudant_hook.get_conn()
        conn = mock_get_connection.return_value
        mock_cloudant.assert_called_once_with(user=conn.login, passwd=conn.password, account=conn.host)
        assert cloudant_session == mock_cloudant.return_value

    @patch('airflow.providers.cloudant.hooks.cloudant.CloudantHook.get_connection', return_value=Connection(login='user'))
    def test_get_conn_invalid_connection(self, mock_get_connection):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AirflowException):
            self.cloudant_hook.get_conn()