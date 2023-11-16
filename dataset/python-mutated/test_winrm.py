from __future__ import annotations
from unittest.mock import patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.microsoft.winrm.hooks.winrm import WinRMHook
pytestmark = pytest.mark.db_test

class TestWinRMHook:

    @patch('airflow.providers.microsoft.winrm.hooks.winrm.Protocol')
    def test_get_conn_exists(self, mock_protocol):
        if False:
            while True:
                i = 10
        winrm_hook = WinRMHook()
        winrm_hook.client = mock_protocol.return_value.open_shell.return_value
        conn = winrm_hook.get_conn()
        assert conn == winrm_hook.client

    def test_get_conn_missing_remote_host(self):
        if False:
            while True:
                i = 10
        with pytest.raises(AirflowException):
            WinRMHook().get_conn()

    @patch('airflow.providers.microsoft.winrm.hooks.winrm.Protocol')
    def test_get_conn_error(self, mock_protocol):
        if False:
            return 10
        mock_protocol.side_effect = Exception('Error')
        with pytest.raises(AirflowException):
            WinRMHook(remote_host='host').get_conn()

    @patch('airflow.providers.microsoft.winrm.hooks.winrm.Protocol', autospec=True)
    @patch('airflow.providers.microsoft.winrm.hooks.winrm.WinRMHook.get_connection', return_value=Connection(login='username', password='password', host='remote_host', extra='{\n                   "endpoint": "endpoint",\n                   "remote_port": 123,\n                   "transport": "plaintext",\n                   "service": "service",\n                   "keytab": "keytab",\n                   "ca_trust_path": "ca_trust_path",\n                   "cert_pem": "cert_pem",\n                   "cert_key_pem": "cert_key_pem",\n                   "server_cert_validation": "validate",\n                   "kerberos_delegation": "true",\n                   "read_timeout_sec": 124,\n                   "operation_timeout_sec": 123,\n                   "kerberos_hostname_override": "kerberos_hostname_override",\n                   "message_encryption": "auto",\n                   "credssp_disable_tlsv1_2": "true",\n                   "send_cbt": "false"\n               }'))
    def test_get_conn_from_connection(self, mock_get_connection, mock_protocol):
        if False:
            while True:
                i = 10
        connection = mock_get_connection.return_value
        winrm_hook = WinRMHook(ssh_conn_id='conn_id')
        winrm_hook.get_conn()
        mock_get_connection.assert_called_once_with(winrm_hook.ssh_conn_id)
        mock_protocol.assert_called_once_with(endpoint=str(connection.extra_dejson['endpoint']), transport=str(connection.extra_dejson['transport']), username=connection.login, password=connection.password, service=str(connection.extra_dejson['service']), keytab=str(connection.extra_dejson['keytab']), ca_trust_path=str(connection.extra_dejson['ca_trust_path']), cert_pem=str(connection.extra_dejson['cert_pem']), cert_key_pem=str(connection.extra_dejson['cert_key_pem']), server_cert_validation=str(connection.extra_dejson['server_cert_validation']), kerberos_delegation=str(connection.extra_dejson['kerberos_delegation']).lower() == 'true', read_timeout_sec=int(connection.extra_dejson['read_timeout_sec']), operation_timeout_sec=int(connection.extra_dejson['operation_timeout_sec']), kerberos_hostname_override=str(connection.extra_dejson['kerberos_hostname_override']), message_encryption=str(connection.extra_dejson['message_encryption']), credssp_disable_tlsv1_2=str(connection.extra_dejson['credssp_disable_tlsv1_2']).lower() == 'true', send_cbt=str(connection.extra_dejson['send_cbt']).lower() == 'true')

    @patch('airflow.providers.microsoft.winrm.hooks.winrm.getuser', return_value='user')
    @patch('airflow.providers.microsoft.winrm.hooks.winrm.Protocol')
    def test_get_conn_no_username(self, mock_protocol, mock_getuser):
        if False:
            while True:
                i = 10
        winrm_hook = WinRMHook(remote_host='host', password='password')
        winrm_hook.get_conn()
        assert mock_getuser.return_value == winrm_hook.username

    @patch('airflow.providers.microsoft.winrm.hooks.winrm.Protocol')
    def test_get_conn_no_endpoint(self, mock_protocol):
        if False:
            print('Hello World!')
        winrm_hook = WinRMHook(remote_host='host', password='password')
        winrm_hook.get_conn()
        assert f'http://{winrm_hook.remote_host}:{winrm_hook.remote_port}/wsman' == winrm_hook.endpoint