from unittest import mock
import pytest
import requests
from json import dumps
from apprise.plugins.NotifyBurstSMS import NotifyBurstSMS
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('burstsms://', {'instance': TypeError}), ('burstsms://:@/', {'instance': TypeError}), ('burstsms://{}@12345678'.format('a' * 8), {'instance': TypeError}), ('burstsms://{}:{}@%20'.format('d' * 8, 'e' * 16), {'instance': TypeError}), ('burstsms://{}:{}@{}/123/{}/abcd/'.format('f' * 8, 'g' * 16, '3' * 11, '9' * 15), {'instance': NotifyBurstSMS, 'notify_response': False, 'privacy_url': 'burstsms://f...f:****@'}), ('burstsms://{}:{}@{}'.format('h' * 8, 'i' * 16, '5' * 11), {'instance': NotifyBurstSMS, 'notify_response': False}), ('burstsms://_?key={}&secret={}&from={}&to={}'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': NotifyBurstSMS}), ('burstsms://_?key={}&secret={}&from={}&to={}&batch=y'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': NotifyBurstSMS}), ('burstsms://_?key={}&secret={}&source={}&to={}&country=us'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': NotifyBurstSMS}), ('burstsms://_?key={}&secret={}&source={}&to={}&country=invalid'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': TypeError}), ('burstsms://_?key={}&secret={}&source={}&to={}&validity=10'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': NotifyBurstSMS}), ('burstsms://_?key={}&secret={}&source={}&to={}&validity=invalid'.format('a' * 8, 'b' * 16, '5' * 11, '6' * 11), {'instance': TypeError}), ('burstsms://_?key={}&secret={}&from={}&to={}'.format('a' * 8, 'b' * 16, '5' * 11, '7' * 11), {'instance': NotifyBurstSMS}), ('burstsms://{}:{}@{}/{}'.format('a' * 8, 'b' * 16, '6' * 11, '7' * 11), {'instance': NotifyBurstSMS, 'response': False, 'requests_response_code': 999}), ('burstsms://{}:{}@{}/{}'.format('a' * 8, 'b' * 16, '6' * 11, '7' * 11), {'instance': NotifyBurstSMS, 'test_requests_exceptions': True}))

def test_plugin_burstsms_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyBurstSMS() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_burstsms_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyBurstSMS() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    apikey = '{}'.format('b' * 8)
    secret = '{}'.format('b' * 16)
    source = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifyBurstSMS(apikey=None, secret=secret, source=source)
    with pytest.raises(TypeError):
        NotifyBurstSMS(apikey='  ', secret=secret, source=source)
    with pytest.raises(TypeError):
        NotifyBurstSMS(apikey=apikey, secret=None, source=source)
    with pytest.raises(TypeError):
        NotifyBurstSMS(apikey=apikey, secret='  ', source=source)
    response.status_code = 400
    response.content = dumps({'error': {'code': 'FIELD_INVALID', 'description': 'Sender ID must be one of the numbers that are currently leased.'}})
    mock_post.return_value = response
    obj = NotifyBurstSMS(apikey=apikey, secret=secret, source=source)
    assert obj.notify('title', 'body', 'info') is False