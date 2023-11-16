import os
from json import loads
from unittest import mock
from inspect import cleandoc
import pytest
import requests
from apprise import Apprise
from apprise.plugins.NotifySignalAPI import NotifySignalAPI
from helpers import AppriseURLTester
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.config.ConfigBase import ConfigBase
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')

@pytest.fixture
def request_mock(mocker):
    if False:
        return 10
    '\n    Prepare requests mock.\n    '
    mock_post = mocker.patch('requests.post')
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = ''
    return mock_post
apprise_url_tests = (('signal://', {'instance': TypeError}), ('signal://:@/', {'instance': TypeError}), ('signal://localhost', {'instance': TypeError}), ('signal://localhost', {'instance': TypeError}), ('signal://localhost/123', {'instance': TypeError}), ('signal://localhost/{}/123/'.format('1' * 11), {'instance': NotifySignalAPI, 'response': False, 'privacy_url': 'signal://localhost/+{}/123'.format('1' * 11)}), ('signal://localhost:8080/{}/'.format('1' * 11), {'instance': NotifySignalAPI}), ('signal://localhost:8082/+{}/@group.abcd/'.format('2' * 11), {'instance': NotifySignalAPI, 'privacy_url': 'signal://localhost:8082/+{}/@abcd'.format('2' * 11)}), ('signal://localhost:8080/+{}/group.abcd/'.format('1' * 11), {'instance': NotifySignalAPI, 'privacy_url': 'signal://localhost:8080/+{}/@abcd'.format('1' * 11)}), ('signal://localhost:8080/?from={}&to={},{}'.format('1' * 11, '2' * 11, '3' * 11), {'instance': NotifySignalAPI}), ('signal://localhost:8080/?from={}&to={},{},{}'.format('1' * 11, '2' * 11, '3' * 11, '5' * 3), {'instance': NotifySignalAPI}), ('signal://localhost:8080/{}/{}/?from={}'.format('1' * 11, '2' * 11, '3' * 11), {'instance': NotifySignalAPI}), ('signals://user@localhost/{}/{}'.format('1' * 11, '3' * 11), {'instance': NotifySignalAPI}), ('signals://user:password@localhost/{}/{}'.format('1' * 11, '3' * 11), {'instance': NotifySignalAPI}), ('signals://user:password@localhost/{}/{}'.format('1' * 11, '3' * 11), {'instance': NotifySignalAPI, 'requests_response_code': 201}), ('signals://localhost/{}/{}/{}?batch=True'.format('1' * 11, '3' * 11, '4' * 11), {'instance': NotifySignalAPI}), ('signals://localhost/{}/{}/{}?status=True'.format('1' * 11, '3' * 11, '4' * 11), {'instance': NotifySignalAPI}), ('signal://localhost/{}/{}'.format('1' * 11, '4' * 11), {'instance': NotifySignalAPI, 'response': False, 'requests_response_code': 999}), ('signal://localhost/{}/{}'.format('1' * 11, '4' * 11), {'instance': NotifySignalAPI, 'test_requests_exceptions': True}))

def test_plugin_signal_urls():
    if False:
        print('Hello World!')
    '\n    NotifySignalAPI() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_signal_edge_cases(request_mock):
    if False:
        i = 10
        return i + 15
    '\n    NotifySignalAPI() Edge Cases\n\n    '
    source = '+1 (555) 123-3456'
    target = '+1 (555) 987-5432'
    body = 'test body'
    title = 'My Title'
    with pytest.raises(TypeError):
        NotifySignalAPI(source=None)
    aobj = Apprise()
    assert aobj.add('signals://localhost:231/{}/{}'.format(source, target))
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 1
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'https://localhost:231/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == 'My Title\r\ntest body'
    request_mock.reset_mock()
    aobj = Apprise()
    assert aobj.add('signals://user@localhost:231/{}/{}?status=True'.format(source, target))
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 1
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'https://localhost:231/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == '[i] My Title\r\ntest body'

def test_plugin_signal_yaml_config(request_mock):
    if False:
        return 10
    '\n    NotifySignalAPI() YAML Configuration\n    '
    (result, config) = ConfigBase.config_parse_yaml(cleandoc('\n    urls:\n      - signal://signal:8080/+1234567890:\n         - to: +0987654321\n           tag: signal\n    '))
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0].tags) == 1
    assert 'signal' in result[0].tags
    plugin = result[0]
    assert len(plugin.targets) == 1
    assert '+1234567890' == plugin.source
    assert '+0987654321' in plugin.targets
    (result, config) = ConfigBase.config_parse_yaml(cleandoc('\n    urls:\n      - signal://signal:8080/+1234567890/+0987654321:\n         - tag: signal\n    '))
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0].tags) == 1
    assert 'signal' in result[0].tags
    plugin = result[0]
    assert len(plugin.targets) == 1
    assert '+1234567890' == plugin.source
    assert '+0987654321' in plugin.targets

def test_plugin_signal_based_on_feedback(request_mock):
    if False:
        i = 10
        return i + 15
    '\n    NotifySignalAPI() User Feedback Test\n\n    '
    body = 'test body'
    title = 'My Title'
    aobj = Apprise()
    aobj.add('signal://10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=yes')
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 1
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == 'My Title\r\ntest body'
    assert payload['number'] == '+12512222222'
    assert len(payload['recipients']) == 2
    assert '+12513333333' in payload['recipients']
    assert '+12514444444' in payload['recipients']
    request_mock.reset_mock()
    aobj = Apprise()
    aobj.add('signal://10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=no')
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 2
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == 'My Title\r\ntest body'
    assert payload['number'] == '+12512222222'
    assert len(payload['recipients']) == 1
    assert '+12513333333' in payload['recipients']
    details = request_mock.call_args_list[1]
    assert details[0][0] == 'http://10.0.0.112:8080/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == 'My Title\r\ntest body'
    assert payload['number'] == '+12512222222'
    assert len(payload['recipients']) == 1
    assert '+12514444444' in payload['recipients']
    request_mock.reset_mock()
    aobj = Apprise()
    aobj.add('signal://10.0.0.112:8080/+12513333333/@group1/@group2/12514444444?batch=yes')
    assert aobj.notify(title=title, body=body)
    assert request_mock.call_count == 1
    details = request_mock.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/v2/send'
    payload = loads(details[1]['data'])
    assert payload['message'] == 'My Title\r\ntest body'
    assert payload['number'] == '+12513333333'
    assert len(payload['recipients']) == 3
    assert '+12514444444' in payload['recipients']
    assert 'group.group1' in payload['recipients']
    assert 'group.group2' in payload['recipients']
    assert '/@group1' in aobj[0].url()
    assert '/@group2' in aobj[0].url()
    assert '/+12514444444' in aobj[0].url()

def test_notify_signal_plugin_attachments(request_mock):
    if False:
        i = 10
        return i + 15
    '\n    NotifySignalAPI() Attachments\n\n    '
    obj = Apprise.instantiate('signal://10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=no')
    assert isinstance(obj, NotifySignalAPI)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    with mock.patch('builtins.open', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('signal://10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=yes')
    assert isinstance(obj, NotifySignalAPI)
    request_mock.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert request_mock.call_count == 1