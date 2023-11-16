import os
import sys
import json
from unittest import mock
import requests
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.plugins.NotifyJSON import NotifyJSON
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('json://:@/', {'instance': None}), ('json://', {'instance': None}), ('jsons://', {'instance': None}), ('json://localhost', {'instance': NotifyJSON}), ('json://user@localhost?method=invalid', {'instance': TypeError}), ('json://user:pass@localhost', {'instance': NotifyJSON, 'privacy_url': 'json://user:****@localhost'}), ('json://user@localhost', {'instance': NotifyJSON}), ('json://user@localhost?method=put', {'instance': NotifyJSON}), ('json://user@localhost?method=get', {'instance': NotifyJSON}), ('json://user@localhost?method=post', {'instance': NotifyJSON}), ('json://user@localhost?method=head', {'instance': NotifyJSON}), ('json://user@localhost?method=delete', {'instance': NotifyJSON}), ('json://user@localhost?method=patch', {'instance': NotifyJSON}), ('json://localhost:8080', {'instance': NotifyJSON}), ('json://user:pass@localhost:8080', {'instance': NotifyJSON}), ('jsons://localhost', {'instance': NotifyJSON}), ('jsons://user:pass@localhost', {'instance': NotifyJSON}), ('jsons://localhost:8080/path/', {'instance': NotifyJSON, 'privacy_url': 'jsons://localhost:8080/path/'}), ('json://localhost:8080/path?-ParamA=Value', {'instance': NotifyJSON}), ('jsons://user:password@localhost:8080', {'instance': NotifyJSON, 'privacy_url': 'jsons://user:****@localhost:8080'}), ('json://localhost:8080/path?+HeaderKey=HeaderValue', {'instance': NotifyJSON}), ('json://user:pass@localhost:8081', {'instance': NotifyJSON, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('json://user:pass@localhost:8082', {'instance': NotifyJSON, 'response': False, 'requests_response_code': 999}), ('json://user:pass@localhost:8083', {'instance': NotifyJSON, 'test_requests_exceptions': True}))

def test_plugin_custom_json_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyJSON() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_custom_json_edge_cases(mock_get, mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyJSON() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    mock_get.return_value = response
    results = NotifyJSON.parse_url('json://localhost:8080/command?:message=msg&:test=value&method=GET&:type=')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8080
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'json'
    assert results['url'] == 'json://localhost:8080/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['message'] == 'msg'
    assert results['qsd:']['type'] == ''
    instance = NotifyJSON(**results)
    assert isinstance(instance, NotifyJSON)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 1
    details = mock_get.call_args_list[0]
    assert details[0][0] == 'http://localhost:8080/command'
    assert 'title' in details[1]['data']
    dataset = json.loads(details[1]['data'])
    assert dataset['title'] == 'title'
    assert 'message' not in dataset
    assert 'msg' in dataset
    assert 'type' not in dataset
    assert dataset['msg'] == 'body'
    assert 'test' in dataset
    assert dataset['test'] == 'value'
    assert instance.url(privacy=False).startswith('json://localhost:8080/command?')
    new_results = NotifyJSON.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'method'):
        assert new_results[k] == results[k]

@mock.patch('requests.post')
def test_notify_json_plugin_attachments(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyJSON() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    obj = Apprise.instantiate('json://localhost.localdomain/')
    assert isinstance(obj, NotifyJSON)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    if sys.version_info.major >= 3:
        builtin_open_function = 'builtins.open'
    else:
        builtin_open_function = '__builtin__.open'
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    mock_post.side_effect = None
    mock_post.return_value = okay_response
    with mock.patch(builtin_open_function, side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('json://no-reply@example.com/')
    assert isinstance(obj, NotifyJSON)
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 1

@mock.patch('requests.post')
def test_plugin_custom_form_for_synology(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyJSON() Synology Chat Test Case\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    results = NotifyJSON.parse_url('jsons://localhost:8081/webapi/entry.cgi?-api=SYNO.Chat.External&-method=incoming&-version=2&-token=abc123&:message=text&:version=&:type=&:title=&:attachments&:file_url=https://i.redd.it/my2t4d2fx0u31.jpg')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8081
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/webapi/entry.cgi'
    assert results['path'] == '/webapi/'
    assert results['query'] == 'entry.cgi'
    assert results['schema'] == 'jsons'
    assert results['url'] == 'jsons://localhost:8081/webapi/entry.cgi'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd-']['api'] == 'SYNO.Chat.External'
    assert results['qsd-']['method'] == 'incoming'
    assert results['qsd-']['version'] == '2'
    assert results['qsd-']['token'] == 'abc123'
    instance = NotifyJSON(**results)
    assert isinstance(instance, NotifyJSON)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'https://localhost:8081/webapi/entry.cgi'
    params = details[1]['params']
    assert params.get('api') == 'SYNO.Chat.External'
    assert params.get('method') == 'incoming'
    assert params.get('version') == '2'
    assert params.get('token') == 'abc123'
    payload = json.loads(details[1]['data'])
    assert 'version' not in payload
    assert 'title' not in payload
    assert 'message' not in payload
    assert 'type' not in payload
    assert payload.get('text') == 'body'
    assert payload.get('file_url') == 'https://i.redd.it/my2t4d2fx0u31.jpg'