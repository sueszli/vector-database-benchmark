from unittest import mock
import pytest
import requests
from json import dumps
from apprise import Apprise
from apprise.plugins.NotifyVoipms import NotifyVoipms
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('voipms://', {'instance': TypeError}), ('voipms://@:', {'instance': TypeError}), ('voipms://{}/{}'.format('user@example.com', '1' * 11), {'instance': TypeError}), ('voipms://:{}'.format('password'), {'instance': TypeError}), ('voipms://{}:{}/{}'.format('user@', 'pass', '1' * 11), {'instance': TypeError}), ('voipms://{password}:{email}'.format(email='user@example.com', password='password'), {'instance': TypeError}), ('voipms://{password}:{email}/1613'.format(email='user@example.com', password='password'), {'instance': TypeError}), ('voipms://{password}:{email}/01133122446688'.format(email='user@example.com', password='password'), {'instance': TypeError}), ('voipms://{password}:{email}/{from_phone}/{targets}/'.format(email='user@example.com', password='password', from_phone='16134448888', targets='/'.join(['26134442222'])), {'instance': NotifyVoipms, 'response': False, 'requests_response_code': 999}), ('voipms://{password}:{email}/{from_phone}'.format(email='user@example.com', password='password', from_phone='16138884444'), {'instance': NotifyVoipms, 'response': False, 'requests_response_code': 999}), ('voipms://{password}:{email}/?from={from_phone}'.format(email='user@example.com', password='password', from_phone='16138884444'), {'instance': NotifyVoipms, 'response': False, 'requests_response_code': 999}), ('voipms://{password}:{email}/{from_phone}/{targets}/'.format(email='user@example.com', password='password', from_phone='16138884444', targets='/'.join(['16134442222'])), {'instance': NotifyVoipms, 'response': True, 'privacy_url': 'voipms://p...d:user@example.com/16...4'}), ('voipms://{password}:{email}/{from_phone}/{targets}/'.format(email='user@example.com', password='password', from_phone='16138884444', targets='/'.join(['16134442222', '16134443333'])), {'instance': NotifyVoipms, 'response': True, 'privacy_url': 'voipms://p...d:user@example.com/16...4'}), ('voipms://{password}:{email}/?from={from_phone}&to={targets}'.format(email='user@example.com', password='password', from_phone='16138884444', targets='16134448888'), {'instance': NotifyVoipms}), ('voipms://{password}:{email}/{from_phone}/{targets}/'.format(email='user@example.com', password='password', from_phone='16138884444', targets='16134442222'), {'instance': NotifyVoipms, 'test_requests_exceptions': True}))

def test_plugin_voipms():
    if False:
        i = 10
        return i + 15
    '\n    NotifyVoipms() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
def test_plugin_voipms_edge_cases(mock_get):
    if False:
        while True:
            i = 10
    '\n    NotifyVoipms() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_get.return_value = response
    email = 'user@example.com'
    password = 'password'
    source = '+1 (555) 123-3456'
    targets = '+1 (555) 123-9876'
    with pytest.raises(TypeError):
        NotifyVoipms(email=None, source=source)
    response.status_code = 400
    response.content = dumps({'code': 21211, 'message': 'Unable to process your request.'})
    mock_get.return_value = response
    obj = Apprise.instantiate('voipms://{password}:{email}/{source}/{targets}'.format(email=email, password=password, source=source, targets=targets))
    assert isinstance(obj, NotifyVoipms)
    assert obj.notify('title', 'body', 'info') is False

@mock.patch('requests.get')
def test_plugin_voipms_non_success_status(mock_get):
    if False:
        i = 10
        return i + 15
    '\n    NotifyVoipms() Non Success Status\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_get.return_value = response
    response.status_code = 200
    response.content = dumps({'status': 'invalid_credentials', 'message': 'Username or Password is incorrect'})
    obj = Apprise.instantiate('voipms://{password}:{email}/{source}/{targets}'.format(email='user@example.com', password='badpassword', source='16134448888', targets='16134442222'))
    assert isinstance(obj, NotifyVoipms)
    assert obj.notify('title', 'body', 'info') is False
    response.content = '{'
    assert obj.send('title', 'body') is False