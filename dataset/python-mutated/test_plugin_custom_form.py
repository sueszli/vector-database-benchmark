import os
from unittest import mock
import requests
from apprise.plugins.NotifyForm import NotifyForm
from helpers import AppriseURLTester
from apprise import Apprise
from apprise import NotifyType
from apprise import AppriseAttachment
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('form://:@/', {'instance': None}), ('form://', {'instance': None}), ('forms://', {'instance': None}), ('form://localhost', {'instance': NotifyForm}), ('form://user@localhost?method=invalid', {'instance': TypeError}), ('form://user:pass@localhost', {'instance': NotifyForm, 'privacy_url': 'form://user:****@localhost'}), ('form://user@localhost', {'instance': NotifyForm}), ('form://user@localhost?method=put', {'instance': NotifyForm}), ('form://user@localhost?method=get', {'instance': NotifyForm}), ('form://user@localhost?method=post', {'instance': NotifyForm}), ('form://user@localhost?method=head', {'instance': NotifyForm}), ('form://user@localhost?method=delete', {'instance': NotifyForm}), ('form://user@localhost?method=patch', {'instance': NotifyForm}), ('form://localhost:8080?:key=value&:key2=value2', {'instance': NotifyForm}), ('form://localhost:8080', {'instance': NotifyForm}), ('form://user:pass@localhost:8080', {'instance': NotifyForm}), ('forms://localhost', {'instance': NotifyForm}), ('forms://user:pass@localhost', {'instance': NotifyForm}), ('forms://localhost:8080/path/', {'instance': NotifyForm, 'privacy_url': 'forms://localhost:8080/path/'}), ('forms://user:password@localhost:8080', {'instance': NotifyForm, 'privacy_url': 'forms://user:****@localhost:8080'}), ('form://localhost:8080/path?-ParamA=Value', {'instance': NotifyForm}), ('form://localhost:8080/path?+HeaderKey=HeaderValue', {'instance': NotifyForm}), ('form://user:pass@localhost:8081', {'instance': NotifyForm, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('form://user:pass@localhost:8082', {'instance': NotifyForm, 'response': False, 'requests_response_code': 999}), ('form://user:pass@localhost:8083', {'instance': NotifyForm, 'test_requests_exceptions': True}))

def test_plugin_custom_form_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyForm() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_custom_form_attachments(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyForm() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    obj = Apprise.instantiate('form://user@localhost.localdomain/?method=post')
    assert isinstance(obj, NotifyForm)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    mock_post.return_value = None
    mock_post.side_effect = OSError()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    mock_post.side_effect = None
    mock_post.return_value = okay_response
    with mock.patch('builtins.open', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    with mock.patch('builtins.open', side_effect=[None, OSError(), None]):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    mock_post.return_value = None
    mock_post.side_effect = OSError()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    mock_post.return_value = okay_response
    mock_post.side_effect = None
    obj = Apprise.instantiate('form://user@localhost.localdomain/?attach-as=file')
    assert isinstance(obj, NotifyForm)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    for attach_as in ('file*', '*file', 'file*file', 'file:', ':file', 'file:file', 'file?', '?file', 'file?file', 'file.', '.file', 'file.file', 'file+', '+file', 'file+file', 'file$', '$file', 'file$file'):
        obj = Apprise.instantiate(f'form://user@localhost.localdomain/?attach-as={attach_as}')
        assert isinstance(obj, NotifyForm)
        path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
        attach = AppriseAttachment(path)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
        path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
        attach = AppriseAttachment(path)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    obj = Apprise.instantiate('form://user@localhost.localdomain/?attach-as={')
    assert obj is None

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_custom_form_edge_cases(mock_get, mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyForm() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    mock_get.return_value = response
    results = NotifyForm.parse_url('form://localhost:8080/command?:message=msg&:abcd=test&method=POST')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8080
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'form'
    assert results['url'] == 'form://localhost:8080/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['abcd'] == 'test'
    assert results['qsd:']['message'] == 'msg'
    instance = NotifyForm(**results)
    assert isinstance(instance, NotifyForm)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 1
    assert mock_get.call_count == 0
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://localhost:8080/command'
    assert 'abcd' in details[1]['data']
    assert details[1]['data']['abcd'] == 'test'
    assert 'title' in details[1]['data']
    assert details[1]['data']['title'] == 'title'
    assert 'message' not in details[1]['data']
    assert 'msg' in details[1]['data']
    assert details[1]['data']['msg'] == 'body'
    assert instance.url(privacy=False).startswith('form://localhost:8080/command?')
    new_results = NotifyForm.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'payload', 'method'):
        assert new_results[k] == results[k]
    mock_post.reset_mock()
    mock_get.reset_mock()
    results = NotifyForm.parse_url('form://localhost:8080/command?:type=&:message=msg&method=POST')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8080
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'form'
    assert results['url'] == 'form://localhost:8080/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['message'] == 'msg'
    instance = NotifyForm(**results)
    assert isinstance(instance, NotifyForm)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 1
    assert mock_get.call_count == 0
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://localhost:8080/command'
    assert 'title' in details[1]['data']
    assert details[1]['data']['title'] == 'title'
    assert 'type' not in details[1]['data']
    assert details[1]['data']['msg'] == 'body'
    assert 'message' not in details[1]['data']
    assert 'msg' in details[1]['data']
    assert details[1]['data']['msg'] == 'body'
    assert instance.url(privacy=False).startswith('form://localhost:8080/command?')
    new_results = NotifyForm.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'payload', 'method'):
        assert new_results[k] == results[k]
    mock_post.reset_mock()
    mock_get.reset_mock()
    results = NotifyForm.parse_url('form://localhost:8080/command?:message=test&method=GET')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8080
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'form'
    assert results['url'] == 'form://localhost:8080/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['message'] == 'test'
    instance = NotifyForm(**results)
    assert isinstance(instance, NotifyForm)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 1
    details = mock_get.call_args_list[0]
    assert details[0][0] == 'http://localhost:8080/command'
    assert 'title' in details[1]['params']
    assert details[1]['params']['title'] == 'title'
    assert 'message' not in details[1]['params']
    assert 'test' in details[1]['params']
    assert details[1]['params']['test'] == 'body'
    assert instance.url(privacy=False).startswith('form://localhost:8080/command?')
    new_results = NotifyForm.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'payload', 'method'):
        assert new_results[k] == results[k]