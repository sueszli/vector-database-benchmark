from __future__ import annotations
from unittest.mock import patch
from airflow.models import Connection
from airflow.providers.cohere.hooks.cohere import CohereHook

class TestCohereHook:
    """
    Test for CohereHook
    """

    def test__get_api_key(self):
        if False:
            return 10
        api_key = 'test'
        api_url = 'http://some_host.com'
        timeout = 150
        max_retries = 5
        with patch.object(CohereHook, 'get_connection', return_value=Connection(conn_type='cohere', password=api_key, host=api_url)), patch('cohere.Client') as client:
            hook = CohereHook(timeout=timeout, max_retries=max_retries)
            _ = hook.get_conn
            client.assert_called_once_with(api_key=api_key, timeout=timeout, max_retries=max_retries, api_url=api_url)