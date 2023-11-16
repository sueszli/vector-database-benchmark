from unittest import mock
import pytest
import requests
from apprise.plugins.NotifyFlock import NotifyFlock
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('flock://', {'instance': TypeError}), ('flock://:@/', {'instance': TypeError}), ('flock://%s' % ('t' * 24), {'instance': NotifyFlock}), ('flock://%s?image=True' % ('t' * 24), {'instance': NotifyFlock, 'privacy_url': 'flock://t...t'}), ('flock://%s?image=False' % ('t' * 24), {'instance': NotifyFlock}), ('flock://%s?image=True' % ('t' * 24), {'instance': NotifyFlock, 'include_image': False}), ('flock://%s?to=u:%s&format=markdown' % ('i' * 24, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s?format=markdown' % ('i' * 24), {'instance': NotifyFlock}), ('flock://%s?format=text' % ('i' * 24), {'instance': NotifyFlock}), ('https://api.flock.com/hooks/sendMessage/{}/'.format('i' * 24), {'instance': NotifyFlock}), ('https://api.flock.com/hooks/sendMessage/{}/?format=markdown'.format('i' * 24), {'instance': NotifyFlock}), ('flock://%s/u:%s?format=markdown' % ('i' * 24, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s/u:%s?format=html' % ('i' * 24, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s/%s?format=text' % ('i' * 24, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s/g:%s/u:%s?format=text' % ('i' * 24, 'g' * 12, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s/#%s/@%s?format=text' % ('i' * 24, 'g' * 12, 'u' * 12), {'instance': NotifyFlock}), ('flock://%s/g:%s/u:%s?format=text' % ('i' * 24, 'g' * 12, 'u' * 10), {'instance': NotifyFlock}), ('flock://%s/g:/u:?format=text' % ('i' * 24), {'instance': TypeError}), ('flock://%s/g:%s/u:%s?format=text' % ('i' * 24, 'g' * 14, 'u' * 10), {'instance': NotifyFlock}), ('flock://%s/g:%s/u:%s?format=text' % ('i' * 24, 'g' * 12, 'u' * 10), {'instance': NotifyFlock, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('flock://%s/' % ('t' * 24), {'instance': NotifyFlock, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('flock://%s/' % ('t' * 24), {'instance': NotifyFlock, 'response': False, 'requests_response_code': 999}), ('flock://%s/' % ('t' * 24), {'instance': NotifyFlock, 'test_requests_exceptions': True}))

def test_plugin_flock_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyFlock() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_flock_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifyFlock() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyFlock(token=None)
    with pytest.raises(TypeError):
        NotifyFlock(token='   ')