from unittest import mock
import requests
from apprise import Apprise
from apprise.plugins.NotifyPushDeer import NotifyPushDeer
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('pushdeer://', {'instance': TypeError}), ('pushdeers://', {'instance': TypeError}), ('pushdeer://localhost/{}'.format('a' * 8), {'instance': NotifyPushDeer, 'response': False, 'requests_response_code': 999}), ('pushdeer://localhost/{}'.format('a' * 8), {'instance': NotifyPushDeer, 'test_requests_exceptions': True}), ('pushdeer://localhost:80/{}'.format('a' * 8), {'instance': NotifyPushDeer, 'response': False, 'requests_response_code': 999}), ('pushdeer://localhost:80/{}'.format('a' * 8), {'instance': NotifyPushDeer, 'test_requests_exceptions': True}), ('pushdeer://{}'.format('a' * 8), {'instance': NotifyPushDeer, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pushdeer://{}'.format('a' * 8), {'instance': NotifyPushDeer, 'test_requests_exceptions': True}))

def test_plugin_pushdeer_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyPushDeer() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_pushdeer_general(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyPushDeer() General Checks\n\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('pushdeer://localhost/pushKey')
    assert isinstance(obj, NotifyPushDeer) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost:80/message/push?pushkey=pushKey'