from unittest import mock
import pytest
import requests
from apprise.plugins.NotifyMessageBird import NotifyMessageBird
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('msgbird://', {'instance': TypeError}), ('msgbird://{}/abcd'.format('a' * 25), {'instance': TypeError}), ('msgbird://{}/123'.format('a' * 25), {'instance': TypeError}), ('msgbird://{}/15551232000'.format('a' * 25), {'instance': NotifyMessageBird, 'privacy_url': 'msgbird://a...a/15551232000'}), ('msgbird://{}/15551232000/abcd'.format('a' * 25), {'instance': NotifyMessageBird, 'notify_response': False}), ('msgbird://{}/15551232000/123'.format('a' * 25), {'instance': NotifyMessageBird, 'notify_response': False}), ('msgbird://{}/?from=15551233000&to=15551232000'.format('a' * 25), {'instance': NotifyMessageBird}), ('msgbird://{}/15551232000'.format('a' * 25), {'instance': NotifyMessageBird, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('msgbird://{}/15551232000'.format('a' * 25), {'instance': NotifyMessageBird, 'response': False, 'requests_response_code': 999}), ('msgbird://{}/15551232000'.format('a' * 25), {'instance': NotifyMessageBird, 'test_requests_exceptions': True}))

def test_plugin_messagebird_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyTemplate() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_messagebird_edge_cases(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMessageBird() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    source = '+1 (555) 123-3456'
    with pytest.raises(TypeError):
        NotifyMessageBird(apikey=None, source=source)
    with pytest.raises(TypeError):
        NotifyMessageBird(apikey='     ', source=source)