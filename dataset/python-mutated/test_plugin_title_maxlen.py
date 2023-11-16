import os
from json import loads
from inspect import cleandoc
import pytest
import requests
from apprise import Apprise
from apprise.config.ConfigBase import ConfigBase
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')

@pytest.fixture
def request_mock(mocker):
    if False:
        i = 10
        return i + 15
    '\n    Prepare requests mock.\n    '
    mock_post = mocker.patch('requests.post')
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = ''
    return mock_post

def test_plugin_title_maxlen(request_mock):
    if False:
        print('Hello World!')
    '\n    plugin title maxlen blending support\n\n    '
    (result, config) = ConfigBase.config_parse_yaml(cleandoc('\n    urls:\n\n      # Our JSON plugin allows for a title definition; we enforce a html format\n      - json://user:pass@example.ca?format=html\n      # Telegram has a title_maxlen of 0\n      - tgram://123456789:AABCeFGhIJKLmnOPqrStUvWxYZ12345678U/987654321\n    '))
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result[0].tags) == 0
    aobj = Apprise()
    aobj.add(result)
    assert len(aobj) == 2
    title = 'Hello World'
    body = 'Foo Bar'
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 2
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'http://example.ca'
    payload = loads(details[1]['data'])
    assert payload['message'] == body
    assert payload['title'] == 'Hello World'
    details = request_mock.call_args_list[1]
    assert details[0][0] == 'https://api.telegram.org/bot123456789:AABCeFGhIJKLmnOPqrStUvWxYZ12345678U/sendMessage'
    payload = loads(details[1]['data'])
    assert payload['text'] == '<b>Hello World</b>\r\nFoo Bar'
    request_mock.reset_mock()
    (result, config) = ConfigBase.config_parse_yaml(cleandoc('\n    urls:\n\n      # Telegram has a title_maxlen of 0\n      - tgram://123456789:AABCeFGhIJKLmnOPqrStUvWxYZ12345678U/987654321\n      # Our JSON plugin allows for a title definition; we enforce a html format\n      - json://user:pass@example.ca?format=html\n    '))
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result[0].tags) == 0
    aobj = Apprise()
    aobj.add(result)
    assert len(aobj) == 2
    title = 'Hello World'
    body = 'Foo Bar'
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 2
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'https://api.telegram.org/bot123456789:AABCeFGhIJKLmnOPqrStUvWxYZ12345678U/sendMessage'
    payload = loads(details[1]['data'])
    assert payload['text'] == '<b>Hello World</b>\r\nFoo Bar'
    details = request_mock.call_args_list[1]
    assert details[0][0] == 'http://example.ca'
    payload = loads(details[1]['data'])
    assert payload['message'] == body
    assert payload['title'] == 'Hello World'