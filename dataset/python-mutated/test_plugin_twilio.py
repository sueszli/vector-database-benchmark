from unittest import mock
import requests
import pytest
from json import dumps
from apprise import Apprise
from apprise.plugins.NotifyTwilio import NotifyTwilio
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('twilio://', {'instance': TypeError}), ('twilio://:@/', {'instance': TypeError}), ('twilio://AC{}@12345678'.format('a' * 32), {'instance': TypeError}), ('twilio://AC{}:{}@_'.format('a' * 32, 'b' * 32), {'instance': TypeError}), ('twilio://AC{}:{}@{}'.format('a' * 32, 'b' * 32, '3' * 5), {'instance': NotifyTwilio, 'notify_response': False}), ('twilio://AC{}:{}@{}'.format('a' * 32, 'b' * 32, '3' * 9), {'instance': TypeError}), ('twilio://AC{}:{}@{}/123/{}/abcd/'.format('a' * 32, 'b' * 32, '3' * 11, '9' * 15), {'instance': NotifyTwilio}), ('twilio://AC{}:{}@12345/{}'.format('a' * 32, 'b' * 32, '4' * 11), {'instance': NotifyTwilio, 'privacy_url': 'twilio://...aaaa:b...b@12345'}), ('twilio://AC{}:{}@123456/{}'.format('a' * 32, 'b' * 32, '4' * 11), {'instance': NotifyTwilio}), ('twilio://AC{}:{}@{}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifyTwilio}), ('twilio://_?sid=AC{}&token={}&from={}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifyTwilio}), ('twilio://_?sid=AC{}&token={}&source={}'.format('a' * 32, 'b' * 32, '5' * 11), {'instance': NotifyTwilio}), ('twilio://_?sid=AC{}&token={}&from={}&to={}'.format('a' * 32, 'b' * 32, '5' * 11, '7' * 13), {'instance': NotifyTwilio}), ('twilio://AC{}:{}@{}'.format('a' * 32, 'b' * 32, '6' * 11), {'instance': NotifyTwilio, 'response': False, 'requests_response_code': 999}), ('twilio://AC{}:{}@{}'.format('a' * 32, 'b' * 32, '6' * 11), {'instance': NotifyTwilio, 'test_requests_exceptions': True}))

def test_plugin_twilio_urls():
    if False:
        return 10
    '\n    NotifyTwilio() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_twilio_auth(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyTwilio() Auth\n      - account-wide auth token\n      - API key and its own auth token\n\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    account_sid = 'AC{}'.format('b' * 32)
    apikey = 'SK{}'.format('b' * 32)
    auth_token = '{}'.format('b' * 32)
    source = '+1 (555) 123-3456'
    dest = '+1 (555) 987-6543'
    message_contents = 'test'
    obj = Apprise.instantiate('twilio://{}:{}@{}/{}'.format(account_sid, auth_token, source, dest))
    assert isinstance(obj, NotifyTwilio) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body=message_contents) is True
    obj = Apprise.instantiate('twilio://{}:{}@{}/{}?apikey={}'.format(account_sid, auth_token, source, dest, apikey))
    assert isinstance(obj, NotifyTwilio) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body=message_contents) is True
    assert mock_post.call_count == 2
    first_call = mock_post.call_args_list[0]
    second_call = mock_post.call_args_list[1]
    assert first_call[0][0] == second_call[0][0] == 'https://api.twilio.com/2010-04-01/Accounts/{}/Messages.json'.format(account_sid)
    assert first_call[1]['data']['Body'] == second_call[1]['data']['Body'] == message_contents
    assert first_call[1]['data']['From'] == second_call[1]['data']['From'] == '+15551233456'
    assert first_call[1]['data']['To'] == second_call[1]['data']['To'] == '+15559876543'
    assert first_call[1]['auth'] == (account_sid, auth_token)
    assert second_call[1]['auth'] == (apikey, auth_token)

@mock.patch('requests.post')
def test_plugin_twilio_edge_cases(mock_post):
    if False:
        return 10
    '\n    NotifyTwilio() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    account_sid = 'AC{}'.format('b' * 32)
    auth_token = '{}'.format('b' * 32)
    source = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifyTwilio(account_sid=None, auth_token=auth_token, source=source)
    with pytest.raises(TypeError):
        NotifyTwilio(account_sid=account_sid, auth_token=None, source=source)
    response.status_code = 400
    response.content = dumps({'code': 21211, 'message': "The 'To' number +1234567 is not a valid phone number."})
    mock_post.return_value = response
    obj = NotifyTwilio(account_sid=account_sid, auth_token=auth_token, source=source)
    assert obj.notify('title', 'body', 'info') is False