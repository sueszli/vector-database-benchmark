import requests
import pytest
from apprise import Apprise
from apprise.plugins.NotifyGoogleChat import NotifyGoogleChat
from helpers import AppriseURLTester
from unittest import mock
from apprise import NotifyType
from json import loads
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('gchat://', {'instance': TypeError}), ('gchat://:@/', {'instance': TypeError}), ('gchat://workspace', {'instance': TypeError}), ('gchat://workspace/key/', {'instance': TypeError}), ('gchat://workspace/key/token', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://w...e/k...y/t...n'}), ('gchat://?workspace=ws&key=mykey&token=mytoken', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://w...s/m...y/m...n'}), ('gchat://?workspace=ws&key=mykey&token=mytoken&thread=abc123', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://w...s/m...y/m...n/a...3'}), ('gchat://?workspace=ws&key=mykey&token=mytoken&threadKey=abc345', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://w...s/m...y/m...n/a...5'}), ('https://chat.googleapis.com/v1/spaces/myworkspace/messages?key=mykey&token=mytoken', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://m...e/m...y/m...n'}), ('https://chat.googleapis.com/v1/spaces/myworkspace/messages?key=mykey&token=mytoken&threadKey=mythreadkey', {'instance': NotifyGoogleChat, 'privacy_url': 'gchat://m...e/m...y/m...n/m...y'}), ('gchat://workspace/key/token', {'instance': NotifyGoogleChat, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('gchat://workspace/key/token', {'instance': NotifyGoogleChat, 'response': False, 'requests_response_code': 999}), ('gchat://workspace/key/token', {'instance': NotifyGoogleChat, 'test_requests_exceptions': True}))

def test_plugin_google_chat_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyGoogleChat() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_google_chat_general(mock_post):
    if False:
        return 10
    '\n    NotifyGoogleChat() General Checks\n\n    '
    workspace = 'ws'
    key = 'key'
    threadkey = 'threadkey'
    token = 'token'
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    obj = Apprise.instantiate('gchat://{}/{}/{}'.format(workspace, key, token))
    assert isinstance(obj, NotifyGoogleChat)
    assert obj.notify(body='test body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://chat.googleapis.com/v1/spaces/ws/messages'
    params = mock_post.call_args_list[0][1]['params']
    assert params.get('token') == token
    assert params.get('key') == key
    assert 'threadKey' not in params
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == 'title\r\ntest body'
    mock_post.reset_mock()
    obj = Apprise.instantiate('gchat://{}/{}/{}/{}'.format(workspace, key, token, threadkey))
    assert isinstance(obj, NotifyGoogleChat)
    assert obj.notify(body='test body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://chat.googleapis.com/v1/spaces/ws/messages'
    params = mock_post.call_args_list[0][1]['params']
    assert params.get('token') == token
    assert params.get('key') == key
    assert params.get('threadKey') == threadkey
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == 'title\r\ntest body'

def test_plugin_google_chat_edge_case():
    if False:
        print('Hello World!')
    '\n    NotifyGoogleChat() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyGoogleChat('workspace', 'webhook', 'token', thread_key=object())