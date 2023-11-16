import pytest
import requests
from apprise.plugins.NotifyPushjet import NotifyPushjet
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('pjet://', {'instance': None}), ('pjets://', {'instance': None}), ('pjet://:@/', {'instance': None}), ('pjet://%s' % ('a' * 32), {'instance': TypeError}), ('pjet://user:pass@localhost/%s' % ('a' * 32), {'instance': NotifyPushjet}), ('pjets://localhost/%s' % ('a' * 32), {'instance': NotifyPushjet}), ('pjet://user:pass@localhost?secret=%s' % ('a' * 32), {'instance': NotifyPushjet, 'privacy_url': 'pjet://user:****@localhost'}), ('pjets://localhost:8080/%s' % ('a' * 32), {'instance': NotifyPushjet}), ('pjets://localhost:8080/%s' % ('a' * 32), {'instance': NotifyPushjet, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pjets://localhost:4343/%s' % ('a' * 32), {'instance': NotifyPushjet, 'response': False, 'requests_response_code': 999}), ('pjet://localhost:8081/%s' % ('a' * 32), {'instance': NotifyPushjet, 'test_requests_exceptions': True}))

def test_plugin_pushjet_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyPushjet() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_pushjet_edge_cases():
    if False:
        return 10
    '\n    NotifyPushjet() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyPushjet(secret_key=None)
    with pytest.raises(TypeError):
        NotifyPushjet(secret_key='  ')