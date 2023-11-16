import sys
from unittest import mock
import pytest
import requests
import json
from apprise import Apprise
from apprise.plugins.NotifySimplePush import NotifySimplePush
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('spush://', {'instance': TypeError}), ('spush://{}'.format('A' * 14), {'instance': NotifySimplePush, 'notify_response': False}), ('spush://{}'.format('Y' * 14), {'instance': NotifySimplePush, 'requests_response_text': {'status': 'OK'}, 'privacy_url': 'spush://Y...Y/'}), ('spush://{}?event=Not%20So%20Good'.format('X' * 14), {'instance': NotifySimplePush, 'requests_response_text': {'status': 'NOT-OK'}, 'notify_response': False}), ('spush://salt:pass@{}'.format('X' * 14), {'instance': NotifySimplePush, 'requests_response_text': {'status': 'OK'}, 'privacy_url': 'spush://****:****@X...X/'}), ('spush://{}'.format('Y' * 14), {'instance': NotifySimplePush, 'response': False, 'requests_response_code': 999, 'requests_response_text': {'status': 'BadRequest', 'message': 'Title or message too long'}}), ('spush://{}'.format('Z' * 14), {'instance': NotifySimplePush, 'test_requests_exceptions': True}))

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_simplepush_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifySimplePush() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@pytest.mark.skipif('cryptography' in sys.modules, reason='Requires that cryptography NOT be installed')
def test_plugin_fcm_cryptography_import_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySimplePush() Cryptography loading failure\n    '
    obj = Apprise.instantiate('spush://{}'.format('Y' * 14))
    assert obj is None

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_simplepush_edge_cases():
    if False:
        while True:
            i = 10
    '\n    NotifySimplePush() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifySimplePush(apikey=None)
    with pytest.raises(TypeError):
        NotifySimplePush(apikey='  ')
    with pytest.raises(TypeError):
        NotifySimplePush(apikey='abc', event=object)
    with pytest.raises(TypeError):
        NotifySimplePush(apikey='abc', event='  ')

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
@mock.patch('requests.post')
def test_plugin_simplepush_general(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySimplePush() General Tests\n    '
    response = mock.Mock()
    response.content = json.dumps({'status': 'OK'})
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('spush://{}'.format('Y' * 14))
    assert obj.notify(title='test', body='test') is True