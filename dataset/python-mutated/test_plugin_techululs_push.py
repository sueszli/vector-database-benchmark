import requests
from apprise.plugins.NotifyTechulusPush import NotifyTechulusPush
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('push://', {'instance': TypeError}), ('push://%s' % ('+' * 24), {'instance': TypeError}), ('push://%s' % UUID4, {'instance': NotifyTechulusPush, 'privacy_url': 'push://8...2/'}), ('push://:@/', {'instance': TypeError}), ('push://%s' % UUID4, {'instance': NotifyTechulusPush, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('push://%s' % UUID4, {'instance': NotifyTechulusPush, 'response': False, 'requests_response_code': 999}), ('push://%s' % UUID4, {'instance': NotifyTechulusPush, 'test_requests_exceptions': True}))

def test_plugin_techulus_push_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyTechulusPush() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()