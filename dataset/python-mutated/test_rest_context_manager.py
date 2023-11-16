import pytest
from azure.core.rest import HttpRequest
from azure.core.exceptions import ResponseNotReadError

def test_normal_call(client, port):
    if False:
        while True:
            i = 10

    def _raise_and_get_text(response):
        if False:
            while True:
                i = 10
        response.raise_for_status()
        assert response.text() == 'Hello, world!'
        assert response.is_closed
    request = HttpRequest('GET', url='/basic/string')
    response = client.send_request(request)
    _raise_and_get_text(response)
    assert response.is_closed
    with client.send_request(request) as response:
        _raise_and_get_text(response)
    response = client.send_request(request)
    with response as response:
        _raise_and_get_text(response)

def test_stream_call(client):
    if False:
        while True:
            i = 10

    def _raise_and_get_text(response):
        if False:
            print('Hello World!')
        response.raise_for_status()
        assert not response.is_closed
        with pytest.raises(ResponseNotReadError):
            response.text()
        response.read()
        assert response.text() == 'Hello, world!'
        assert response.is_closed
    request = HttpRequest('GET', url='/streams/basic')
    response = client.send_request(request, stream=True)
    _raise_and_get_text(response)
    assert response.is_closed
    with client.send_request(request, stream=True) as response:
        _raise_and_get_text(response)
    assert response.is_closed
    response = client.send_request(request, stream=True)
    with response as response:
        _raise_and_get_text(response)