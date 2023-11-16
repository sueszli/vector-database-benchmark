import os
import pytest
import requests
from integration_tests.helpers.network_helpers import get_network_host

@pytest.mark.benchmark
def test_default_url_index_request(default_session):
    if False:
        i = 10
        return i + 15
    BASE_URL = 'http://127.0.0.1:8080'
    res = requests.get(f'{BASE_URL}')
    assert res.status_code == 200

@pytest.mark.benchmark
def test_local_index_request(session):
    if False:
        print('Hello World!')
    BASE_URL = 'http://127.0.0.1:8080'
    res = requests.get(f'{BASE_URL}')
    assert os.getenv('ROBYN_HOST') == '127.0.0.1'
    assert res.status_code == 200

@pytest.mark.benchmark
def test_global_index_request(global_session):
    if False:
        print('Hello World!')
    host = get_network_host()
    BASE_URL = f'http://{host}:8080'
    res = requests.get(f'{BASE_URL}')
    assert os.getenv('ROBYN_HOST') == f'{host}'
    assert res.status_code == 200

@pytest.mark.benchmark
def test_dev_index_request(dev_session):
    if False:
        while True:
            i = 10
    BASE_URL = 'http://127.0.0.1:8081'
    res = requests.get(f'{BASE_URL}')
    assert os.getenv('ROBYN_PORT') == '8081'
    assert res.status_code == 200