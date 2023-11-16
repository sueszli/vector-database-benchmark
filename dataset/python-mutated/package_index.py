"""Hook for additional Package Indexes (Python)."""
from __future__ import annotations
import subprocess
from typing import Any
from urllib.parse import quote, urlparse
from airflow.hooks.base import BaseHook

class PackageIndexHook(BaseHook):
    """Specify package indexes/Python package sources using Airflow connections."""
    conn_name_attr = 'pi_conn_id'
    default_conn_name = 'package_index_default'
    conn_type = 'package_index'
    hook_name = 'Package Index (Python)'

    def __init__(self, pi_conn_id: str=default_conn_name) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.pi_conn_id = pi_conn_id
        self.conn = None

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Return custom field behaviour.'
        return {'hidden_fields': ['schema', 'port', 'extra'], 'relabeling': {'host': 'Package Index URL'}, 'placeholders': {'host': 'Example: https://my-package-mirror.net/pypi/repo-name/simple', 'login': 'Username for package index', 'password': 'Password for package index (will be masked)'}}

    @staticmethod
    def _get_basic_auth_conn_url(index_url: str, user: str | None, password: str | None) -> str:
        if False:
            i = 10
            return i + 15
        'Return a connection URL with basic auth credentials based on connection config.'
        url = urlparse(index_url)
        host = url.netloc.split('@')[-1]
        if user:
            if password:
                host = f'{quote(user)}:{quote(password)}@{host}'
            else:
                host = f'{quote(user)}@{host}'
        return url._replace(netloc=host).geturl()

    def get_conn(self) -> Any:
        if False:
            return 10
        'Return connection for the hook.'
        return self.get_connection_url()

    def get_connection_url(self) -> Any:
        if False:
            print('Hello World!')
        'Return a connection URL with embedded credentials.'
        conn = self.get_connection(self.pi_conn_id)
        index_url = conn.host
        if not index_url:
            raise Exception('Please provide an index URL.')
        return self._get_basic_auth_conn_url(index_url, conn.login, conn.password)

    def test_connection(self) -> tuple[bool, str]:
        if False:
            while True:
                i = 10
        'Test connection to package index url.'
        conn_url = self.get_connection_url()
        proc = subprocess.run(['pip', 'search', 'not-existing-test-package', '--no-input', '--index', conn_url], check=False, capture_output=True)
        conn = self.get_connection(self.pi_conn_id)
        if proc.returncode not in [0, 23]:
            return (False, f'Connection test to {conn.host} failed. Error: {str(proc.stderr)}')
        return (True, f'Connection to {conn.host} tested successfully!')