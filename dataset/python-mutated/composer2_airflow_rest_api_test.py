import os
from unittest import mock
import pytest
import requests
import composer2_airflow_rest_api
COMPOSER2_WEB_SERVER_URL = os.environ['COMPOSER2_WEB_SERVER_URL']
DAG_CONFIG = {'test': 'value'}

@pytest.fixture(scope='function')
def successful_response() -> None:
    if False:
        for i in range(10):
            print('nop')
    response_mock = mock.create_autospec(requests.Response, instance=True)
    response_mock.status_code = 200
    response_mock.text = '"state": "running"'
    response_mock.headers = {'Content-Type': 'text/html; charset=utf-8'}
    with mock.patch('composer2_airflow_rest_api.make_composer2_web_server_request', autospec=True, return_value=response_mock):
        yield

@pytest.fixture(scope='function')
def insufficient_permissions_response() -> None:
    if False:
        print('Hello World!')
    response_mock = mock.create_autospec(requests.Response, instance=True)
    response_mock.status_code = 403
    response_mock.text = 'Mocked insufficient permissions'
    response_mock.headers = {'Content-Type': 'text/html; charset=utf-8'}
    with mock.patch('composer2_airflow_rest_api.make_composer2_web_server_request', autospec=True, return_value=response_mock):
        yield

def test_trigger_dag_insufficient_permissions(insufficient_permissions_response: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(requests.HTTPError, match='You do not have a permission to perform this operation.'):
        composer2_airflow_rest_api.trigger_dag(COMPOSER2_WEB_SERVER_URL, 'airflow_monitoring', DAG_CONFIG)

def test_trigger_dag_incorrect_environment() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(requests.HTTPError, match='404 Client Error: Not Found for url'):
        composer2_airflow_rest_api.trigger_dag('https://invalid-environment.composer.googleusercontent.com', 'airflow_monitoring', DAG_CONFIG)

def test_trigger_dag(successful_response: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    out = composer2_airflow_rest_api.trigger_dag(COMPOSER2_WEB_SERVER_URL, 'airflow_monitoring', DAG_CONFIG)
    assert '"state": "running"' in out