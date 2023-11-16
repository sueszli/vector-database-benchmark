from websocket import create_connection
import pytest
from integration_tests.helpers.http_methods_helpers import generic_http_helper, head

@pytest.mark.parametrize('http_method_type', ['get', 'post', 'put', 'delete', 'patch', 'options', 'trace'])
@pytest.mark.benchmark
def test_sub_router(http_method_type, session):
    if False:
        return 10
    response = generic_http_helper(http_method_type, 'sub_router/foo')
    assert response.json() == {'message': 'foo'}

@pytest.mark.benchmark
def test_sub_router_head(session):
    if False:
        i = 10
        return i + 15
    response = head('sub_router/foo')
    assert response.text == ''

@pytest.mark.benchmark
def test_sub_router_web_socket(session):
    if False:
        return 10
    BASE_URL = 'ws://127.0.0.1:8080'
    ws = create_connection(f'{BASE_URL}/sub_router/ws')
    assert ws.recv() == 'Hello world, from ws'
    ws.send('My name is?')
    assert ws.recv() == 'Message'