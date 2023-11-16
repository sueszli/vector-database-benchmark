import json
from unittest.mock import patch
import pytest
import requests
from source_hubspot.errors import HubspotInvalidAuth
from source_hubspot.streams import Stream

def mock_retry_func(*args, **kwargs):
    if False:
        print('Hello World!')
    error_message = 'Token expired'
    response = requests.Response()
    response.status_code = 401
    response._content = json.dumps({'message': error_message}).encode()
    raise HubspotInvalidAuth(error_message, response=response)

@patch.multiple(Stream, __abstractmethods__=set())
def test_handle_request_with_retry(common_params):
    if False:
        i = 10
        return i + 15
    stream_instance = Stream(**common_params)
    mock_response = requests.Response()
    mock_response.status_code = 200
    mock_response._content = json.dumps({'data': 'Mocked response'}).encode()
    with patch.object(stream_instance, '_send_request', return_value=mock_response):
        response = stream_instance.handle_request()
    assert response.status_code == 200
    assert response.json() == {'data': 'Mocked response'}

@patch.multiple(Stream, __abstractmethods__=set())
def test_handle_request_with_retry_token_expired(common_params):
    if False:
        i = 10
        return i + 15
    stream_instance = Stream(**common_params)
    with patch.object(stream_instance, '_send_request', side_effect=mock_retry_func) as mocked_send_request:
        with pytest.raises(HubspotInvalidAuth):
            stream_instance.handle_request()
    assert mocked_send_request.call_count == 5