import pytest
import requests
from apprise.plugins.NotifyRyver import NotifyRyver
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('ryver://', {'instance': TypeError}), ('ryver://:@/', {'instance': TypeError}), ('ryver://apprise', {'instance': TypeError}), ('ryver://apprise/ckhrjW8w672m6HG?mode=invalid', {'instance': TypeError}), ('ryver://x/ckhrjW8w672m6HG?mode=slack', {'instance': TypeError}), ('ryver://apprise/ckhrjW8w672m6HG?mode=slack', {'instance': NotifyRyver}), ('ryver://apprise/ckhrjW8w672m6HG?mode=ryver', {'instance': NotifyRyver}), ('ryver://apprise/ckhrjW8w672m6HG?webhook=slack', {'instance': NotifyRyver}), ('ryver://apprise/ckhrjW8w672m6HG?webhook=ryver', {'instance': NotifyRyver, 'privacy_url': 'ryver://apprise/c...G'}), ('https://apprise.ryver.com/application/webhook/ckhrjW8w672m6HG', {'instance': NotifyRyver}), ('https://apprise.ryver.com/application/webhook/ckhrjW8w672m6HG?webhook=ryver', {'instance': NotifyRyver}), ('ryver://caronc@apprise/ckhrjW8w672m6HG', {'instance': NotifyRyver, 'include_image': False}), ('ryver://apprise/ckhrjW8w672m6HG', {'instance': NotifyRyver, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('ryver://apprise/ckhrjW8w672m6HG', {'instance': NotifyRyver, 'response': False, 'requests_response_code': 999}), ('ryver://apprise/ckhrjW8w672m6HG', {'instance': NotifyRyver, 'test_requests_exceptions': True}))

def test_plugin_ryver_urls():
    if False:
        return 10
    '\n    NotifyRyver() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_ryver_edge_cases():
    if False:
        print('Hello World!')
    '\n    NotifyRyver() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyRyver(organization='abc', token=None)
    with pytest.raises(TypeError):
        NotifyRyver(organization='abc', token='  ')
    with pytest.raises(TypeError):
        NotifyRyver(organization=None, token='abc')
    with pytest.raises(TypeError):
        NotifyRyver(organization='  ', token='abc')