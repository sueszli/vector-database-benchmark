from __future__ import annotations
from airflow_breeze.utils import host_info_utils
SUPPORTED_OS = ['linux', 'darwin', 'windows']

def test_get_host_os():
    if False:
        i = 10
        return i + 15
    current_os = host_info_utils.get_host_os()
    assert current_os in SUPPORTED_OS