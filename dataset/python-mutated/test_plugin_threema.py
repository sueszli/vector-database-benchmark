from unittest import mock
import pytest
import requests
from apprise.plugins.NotifyThreema import NotifyThreema
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('threema://', {'instance': TypeError}), ('threema://@:', {'instance': TypeError}), ('threema://user@secret', {'instance': TypeError}), ('threema://*THEGWID@secret/{targets}/'.format(targets='/'.join(['2222'])), {'instance': NotifyThreema, 'notify_response': False, 'privacy_url': 'threema://%2ATHEGWID@****/2222'}), ('threema://*THEGWID@secret/{targets}/'.format(targets='/'.join(['16134442222'])), {'instance': NotifyThreema, 'privacy_url': 'threema://%2ATHEGWID@****/16134442222'}), ('threema://*THEGWID@secret/{targets}/'.format(targets='/'.join(['16134442222', '16134443333'])), {'instance': NotifyThreema, 'privacy_url': 'threema://%2ATHEGWID@****/16134442222/16134443333'}), ('threema:///?secret=secret&from=*THEGWID&to={targets}'.format(targets=','.join(['16134448888', 'user1@gmail.com', 'abcd1234'])), {'instance': NotifyThreema}), ('threema:///?secret=secret&gwid=*THEGWID&to={targets}'.format(targets=','.join(['16134448888', 'user2@gmail.com', 'abcd1234'])), {'instance': NotifyThreema}), ('threema://*THEGWID@secret', {'instance': NotifyThreema, 'notify_response': False}), ('threema://*THEGWID@secret/16134443333', {'instance': NotifyThreema, 'response': False, 'requests_response_code': 999}), ('threema://*THEGWID@secret/16134443333', {'instance': NotifyThreema, 'test_requests_exceptions': True}))

def test_plugin_threema():
    if False:
        while True:
            i = 10
    '\n    NotifyThreema() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_threema_edge_cases(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyThreema() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    gwid = '*THEGWID'
    secret = 'mysecret'
    targets = '+1 (555) 123-9876'
    with pytest.raises(TypeError):
        NotifyThreema(user=gwid, secret=None, targets=targets)
    results = NotifyThreema.parse_url(f'threema://?gwid={gwid}&secret={secret}&to={targets}')
    assert isinstance(results, dict)
    assert results['user'] == gwid
    assert results['secret'] == secret
    assert results['password'] is None
    assert results['port'] is None
    assert results['host'] == ''
    assert results['fullpath'] == '/'
    assert results['path'] == '/'
    assert results['query'] is None
    assert results['schema'] == 'threema'
    assert results['url'] == 'threema:///'
    assert isinstance(results['targets'], list) is True
    assert len(results['targets']) == 1
    assert results['targets'][0] == '+1 (555) 123-9876'
    instance = NotifyThreema(**results)
    assert len(instance.targets) == 1
    assert instance.targets[0] == ('phone', '15551239876')
    assert isinstance(instance, NotifyThreema)
    response = instance.send(title='title', body='body 😊')
    assert response is True
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'https://msgapi.threema.ch/send_simple'
    assert details[1]['headers']['User-Agent'] == 'Apprise'
    assert details[1]['headers']['Accept'] == '*/*'
    assert details[1]['headers']['Content-Type'] == 'application/x-www-form-urlencoded; charset=utf-8'
    assert details[1]['params']['secret'] == secret
    assert details[1]['params']['from'] == gwid
    assert details[1]['params']['phone'] == '15551239876'
    assert details[1]['params']['text'] == 'body 😊'.encode('utf-8')