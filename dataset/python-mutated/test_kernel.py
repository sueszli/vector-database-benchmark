from __future__ import annotations
from unittest.mock import patch
from airflow.models import Connection
from airflow.providers.papermill.hooks.kernel import KernelHook

class TestKernelHook:
    """
    Tests for Kernel connection
    """

    def test_kernel_connection(self):
        if False:
            print('Hello World!')
        '\n        Test that fetches kernelConnection with configured host and ports\n        '
        conn = Connection(conn_type='jupyter_kernel', host='test_host', extra='{"shell_port": 60000, "session_key": "key"}')
        with patch.object(KernelHook, 'get_connection', return_value=conn):
            hook = KernelHook()
        assert hook.get_conn().ip == 'test_host'
        assert hook.get_conn().shell_port == 60000
        assert hook.get_conn().session_key == 'key'