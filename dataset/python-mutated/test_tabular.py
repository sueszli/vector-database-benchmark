from __future__ import annotations
import pytest
import requests_mock
from airflow.providers.tabular.hooks.tabular import TabularHook
pytestmark = pytest.mark.db_test

def test_tabular_hook():
    if False:
        i = 10
        return i + 15
    access_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSU'
    with requests_mock.Mocker() as m:
        m.post('https://api.tabulardata.io/ws/v1/oauth/tokens', json={'access_token': access_token, 'token_type': 'Bearer', 'expires_in': 86400, 'warehouse_id': 'fadc4c31-e81f-48cd-9ce8-64cd5ce3fa5d', 'region': 'us-west-2', 'catalog_url': 'warehouses/fadc4c31-e81f-48cd-9ce8-64cd5ce3fa5d'})
        assert TabularHook().get_conn() == access_token