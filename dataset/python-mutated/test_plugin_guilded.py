import os
from unittest import mock
import pytest
import requests
from apprise.plugins.NotifyGuilded import NotifyGuilded
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('guilded://', {'instance': TypeError}), ('guilded://:@/', {'instance': TypeError}), ('guilded://%s' % ('i' * 24), {'instance': TypeError}), ('guilded://%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://l2g@%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?format=markdown&footer=Yes&image=Yes' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content, 'include_image': False}), ('guilded://%s/%s?format=markdown&footer=Yes&image=No&fields=no' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?format=markdown&footer=Yes&image=Yes' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('https://media.guilded.gg/webhooks/{}/{}'.format('0' * 10, 'B' * 40), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?format=markdown&avatar=No&footer=No' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?format=markdown' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?format=text' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s?avatar_url=http://localhost/test.jpg' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content}), ('guilded://%s/%s' % ('i' * 24, 't' * 64), {'instance': NotifyGuilded, 'requests_response_code': requests.codes.no_content, 'include_image': False}), ('guilded://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyGuilded, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('guilded://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyGuilded, 'response': False, 'requests_response_code': 999}), ('guilded://%s/%s/' % ('a' * 24, 'b' * 64), {'instance': NotifyGuilded, 'test_requests_exceptions': True}))

def test_plugin_guilded_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyGuilded() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_guilded_general(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyGuilded() General Checks\n\n    '
    webhook_id = 'A' * 24
    webhook_token = 'B' * 64
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyGuilded(webhook_id=None, webhook_token=webhook_token)
    with pytest.raises(TypeError):
        NotifyGuilded(webhook_id='  ', webhook_token=webhook_token)
    with pytest.raises(TypeError):
        NotifyGuilded(webhook_id=webhook_id, webhook_token=None)
    with pytest.raises(TypeError):
        NotifyGuilded(webhook_id=webhook_id, webhook_token='   ')
    obj = NotifyGuilded(webhook_id=webhook_id, webhook_token=webhook_token, footer=True, thumbnail=False)
    assert isinstance(obj.url(), str) is True