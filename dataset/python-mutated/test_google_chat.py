"""
Test the Google Chat Execution module.
"""
import pytest
import salt.modules.google_chat as gchat
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {gchat: {}}

def mocked_http_query(url, method, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Mocked data for test_send_message_success\n    '
    return {'status': 200, 'dict': None}

def mocked_http_query_failure(url, method, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Mocked data for test_send_message_failure\n    '
    return {'status': 522, 'dict': None}

def test_send_message_success():
    if False:
        return 10
    '\n    Testing a successful message\n    '
    with patch.dict(gchat.__utils__, {'http.query': mocked_http_query}):
        assert gchat.send_message('https://example.com', 'Yupiii')

def test_send_message_failure():
    if False:
        print('Hello World!')
    '\n    Testing a failed message\n    '
    with patch.dict(gchat.__utils__, {'http.query': mocked_http_query_failure}):
        assert not gchat.send_message('https://example.com', 'Yupiii')