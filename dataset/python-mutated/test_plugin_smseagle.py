import os
from json import loads, dumps
from unittest import mock
import requests
from apprise import Apprise
from apprise.plugins.NotifySMSEagle import NotifySMSEagle
from helpers import AppriseURLTester
from apprise import AppriseAttachment
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
SMSEAGLE_GOOD_RESPONSE = dumps({'result': {'message_id': '748', 'status': 'ok'}})
SMSEAGLE_BAD_RESPONSE = dumps({'result': {'error_text': 'Wrong parameters', 'status': 'error'}})
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('smseagle://', {'instance': TypeError}), ('smseagle://:@/', {'instance': TypeError}), ('smseagle://localhost', {'instance': TypeError}), ('smseagle://%20@localhost', {'instance': TypeError}), ('smseagle://token@localhost/123/', {'instance': NotifySMSEagle, 'response': False, 'privacy_url': 'smseagle://****@localhost/@123', 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/%20/%20/', {'instance': NotifySMSEagle, 'response': False, 'privacy_url': 'smseagle://****@localhost/', 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost:8080/{}/'.format('1' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://localhost:8080/{}/?token=abc1234'.format('1' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/@user/?priority=high', {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/@user/?priority=1', {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/@user/?priority=invalid', {'instance': TypeError}), ('smseagle://token@localhost/@user/?priority=25', {'instance': TypeError}), ('smseagle://token@localhost:8082/#abcd/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagle://****@localhost:8082/#abcd', 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost:8082/@abcd/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagle://****@localhost:8082/@abcd', 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost:8081/contact/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagles://****@localhost:8081/@contact', 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost:8082/@/#/,/', {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE, 'response': False}), ('smseagle://token@localhost:8083/@user/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagle://****@localhost:8083/@user', 'requests_response_text': SMSEAGLE_BAD_RESPONSE, 'response': False}), ('smseagle://token@localhost:8084/@user/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagle://****@localhost:8084/@user', 'requests_response_text': None, 'response': False}), ('smseagle://token@localhost:8085/@user/', {'instance': NotifySMSEagle, 'privacy_url': 'smseagle://****@localhost:8085/@user', 'requests_response_text': '{', 'response': False}), ('smseagle://token@localhost:8086/?to={},{}'.format('2' * 11, '3' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost:8087/?to={},{},{}'.format('2' * 11, '3' * 11, '5' * 3), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost:8088/{}/{}/'.format('2' * 11, '3' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost/{}'.format('3' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost/{}/{}?batch=True'.format('3' * 11, '4' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost/{}/?flash=yes'.format('3' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost/{}/?test=yes'.format('3' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagles://token@localhost/{}/{}?status=True'.format('3' * 11, '4' * 11), {'instance': NotifySMSEagle, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/{}'.format('4' * 11), {'instance': NotifySMSEagle, 'response': False, 'requests_response_code': 999, 'requests_response_text': SMSEAGLE_GOOD_RESPONSE}), ('smseagle://token@localhost/{}'.format('4' * 11), {'instance': NotifySMSEagle, 'test_requests_exceptions': True}))

def test_plugin_smseagle_urls():
    if False:
        while True:
            i = 10
    '\n    NotifySMSEagle() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_smseagle_edge_cases(mock_post):
    if False:
        return 10
    '\n    NotifySMSEagle() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    response.content = SMSEAGLE_GOOD_RESPONSE
    mock_post.return_value = response
    target = '+1 (555) 987-5432'
    body = 'test body'
    title = 'My Title'
    aobj = Apprise()
    assert aobj.add('smseagles://token@localhost:231/{}'.format(target))
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body)
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'https://localhost:231/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['params']['message'] == 'My Title\r\ntest body'
    mock_post.reset_mock()
    aobj = Apprise()
    assert aobj.add('smseagles://token@localhost:231/{}?status=Yes'.format(target))
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body)
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'https://localhost:231/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['params']['message'] == '[i] My Title\r\ntest body'

@mock.patch('requests.post')
def test_plugin_smseagle_result_set(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifySMSEagle() Result Sets\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    response.content = SMSEAGLE_GOOD_RESPONSE
    mock_post.return_value = response
    body = 'test body'
    title = 'My Title'
    aobj = Apprise()
    aobj.add('smseagle://token@10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=yes')
    assert len(aobj[0]) == 1
    assert aobj.notify(title=title, body=body)
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert 'method' in payload
    assert payload['method'] == 'sms.send_sms'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'to' in params
    assert len(params['to'].split(',')) == 3
    assert '+12512222222' in params['to'].split(',')
    assert '+12513333333' in params['to'].split(',')
    assert '12514444444' in params['to'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('smseagle://token@10.0.0.112:8080/#group/Contact/123456789?batch=no')
    assert len(aobj[0]) == 3
    assert aobj.notify(title=title, body=body)
    assert mock_post.call_count == 3
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_sms'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'to' in params
    assert len(params['to'].split(',')) == 1
    assert '123456789' in params['to'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    details = mock_post.call_args_list[1]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_togroup'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'groupname' in params
    assert len(params['groupname'].split(',')) == 1
    assert 'group' in params['groupname'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    details = mock_post.call_args_list[2]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_tocontact'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'contactname' in params
    assert len(params['contactname'].split(',')) == 1
    assert 'Contact' in params['contactname'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('smseagle://token@10.0.0.112:8080/513333333/#group1/@contact1/contact2/12514444444?batch=yes')
    assert len(aobj[0]) == 3
    assert aobj.notify(title=title, body=body)
    assert mock_post.call_count == 3
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_sms'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'to' in params
    assert len(params['to'].split(',')) == 2
    assert '513333333' in params['to'].split(',')
    assert '12514444444' in params['to'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    details = mock_post.call_args_list[1]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_togroup'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'groupname' in params
    assert len(params['groupname'].split(',')) == 1
    assert 'group1' in params['groupname'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    details = mock_post.call_args_list[2]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_tocontact'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'contactname' in params
    assert len(params['contactname'].split(',')) == 2
    assert 'contact1' in params['contactname'].split(',')
    assert 'contact2' in params['contactname'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'My Title\r\ntest body'
    assert '/@contact1' in aobj[0].url()
    assert '/@contact2' in aobj[0].url()
    assert '/#group1' in aobj[0].url()
    assert '/513333333' in aobj[0].url()
    assert '/12514444444' in aobj[0].url()

@mock.patch('requests.post')
def test_notify_smseagle_plugin_result_list(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySMSEagle() Result List Response\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = dumps({'result': [{'message_id': '748', 'status': 'ok'}]})
    mock_post.return_value = okay_response
    obj = Apprise.instantiate('smseagle://token@127.0.0.1/12222222/')
    assert isinstance(obj, NotifySMSEagle)
    assert obj.notify('test') is True
    okay_response.content = dumps({'result': [{'message_id': '748', 'status': 'ok'}, {'message_id': '749', 'status': 'error'}]})
    mock_post.return_value = okay_response
    assert obj.notify('test') is False

@mock.patch('requests.post')
def test_notify_smseagle_plugin_attachments(mock_post):
    if False:
        return 10
    '\n    NotifySMSEagle() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = SMSEAGLE_GOOD_RESPONSE
    mock_post.return_value = okay_response
    obj = Apprise.instantiate('smseagle://token@10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=no')
    assert isinstance(obj, NotifySMSEagle)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    mock_post.side_effect = None
    mock_post.return_value = okay_response
    with mock.patch('builtins.open', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('smseagle://token@10.0.0.112:8080/+12512222222/+12513333333/12514444444?batch=yes')
    assert isinstance(obj, NotifySMSEagle)
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_sms'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'to' in params
    assert len(params['to'].split(',')) == 3
    assert '+12512222222' in params['to'].split(',')
    assert '+12513333333' in params['to'].split(',')
    assert '12514444444' in params['to'].split(',')
    assert params.get('message_type') == 'mms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'title\r\nbody'
    assert 'attachments' in params
    assert isinstance(params['attachments'], list)
    assert len(params['attachments']) == 3
    for entry in params['attachments']:
        assert 'content' in entry
        assert 'content_type' in entry
        assert entry.get('content_type').startswith('image/')
    mock_post.reset_mock()
    obj = Apprise.instantiate('smseagle://token@10.0.0.112:8080/513333333/')
    assert isinstance(obj, NotifySMSEagle)
    attach = os.path.join(TEST_VAR_DIR, 'apprise-test.mp4')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 1
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://10.0.0.112:8080/jsonrpc/sms'
    payload = loads(details[1]['data'])
    assert payload['method'] == 'sms.send_sms'
    assert 'params' in payload
    assert isinstance(payload['params'], dict)
    params = payload['params']
    assert 'to' in params
    assert len(params['to'].split(',')) == 1
    assert '513333333' in params['to'].split(',')
    assert params.get('message_type') == 'sms'
    assert params.get('responsetype') == 'extended'
    assert params.get('access_token') == 'token'
    assert params.get('highpriority') == 0
    assert params.get('flash') == 0
    assert params.get('test') == 0
    assert params.get('unicode') == 1
    assert params.get('message') == 'title\r\nbody'
    assert 'attachments' not in params