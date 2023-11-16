import json
from unittest import mock

def mock_response(status_code=200, headers=None, json_payload=None):
    if False:
        for i in range(10):
            print('nop')
    response = mock.Mock(status_code=status_code, headers=headers or {})
    if json_payload is not None:
        response.text = lambda encoding=None: json.dumps(json_payload)
        response.headers['content-type'] = 'application/json'
        response.content_type = 'application/json'
    else:
        response.text = lambda encoding=None: ''
        response.headers['content-type'] = 'text/plain'
        response.content_type = 'text/plain'
    return response