from unittest import mock
import pytest
import requests
from apprise.plugins.NotifyPushed import NotifyPushed
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('pushed://', {'instance': TypeError}), ('pushed://%s' % ('a' * 32), {'instance': TypeError}), ('pushed://:@/', {'instance': TypeError}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s/#channel/' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s?to=channel' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'privacy_url': 'pushed://a...a/****/'}), ('pushed://%s/%s/dropped_value/' % ('a' * 32, 'a' * 64), {'instance': TypeError}), ('pushed://%s/%s/#channel1/#channel2' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s/@ABCD/' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s/@ABCD/@DEFG/' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s/@ABCD/#channel' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': 999}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'test_requests_exceptions': True}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': 999}), ('pushed://%s/%s/#channel' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': 999}), ('pushed://%s/%s/@user' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'response': False, 'requests_response_code': 999}), ('pushed://%s/%s' % ('a' * 32, 'a' * 64), {'instance': NotifyPushed, 'test_requests_exceptions': True}))

def test_plugin_pushed_urls():
    if False:
        return 10
    '\n    NotifyPushed() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_pushed_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifyPushed() Edge Cases\n\n    '
    recipients = '@ABCDEFG, @DEFGHIJ, #channel, #channel2'
    app_key = 'ABCDEFG'
    app_secret = 'ABCDEFG'
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyPushed(app_key=None, app_secret=app_secret, recipients=None)
    with pytest.raises(TypeError):
        NotifyPushed(app_key='  ', app_secret=app_secret, recipients=None)
    with pytest.raises(TypeError):
        NotifyPushed(app_key=app_key, app_secret=None, recipients=None)
    with pytest.raises(TypeError):
        NotifyPushed(app_key=app_key, app_secret='   ')
    obj = NotifyPushed(app_key=app_key, app_secret=app_secret, recipients=None)
    assert isinstance(obj, NotifyPushed) is True
    assert len(obj.channels) == 0
    assert len(obj.users) == 0
    obj = NotifyPushed(app_key=app_key, app_secret=app_secret, targets=recipients)
    assert isinstance(obj, NotifyPushed) is True
    assert len(obj.channels) == 2
    assert len(obj.users) == 2
    mock_post.return_value.status_code = requests.codes.internal_server_error
    mock_get.return_value.status_code = requests.codes.internal_server_error