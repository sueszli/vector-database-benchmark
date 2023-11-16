import requests
from apprise.plugins.NotifyFaast import NotifyFaast
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('faast://', {'instance': TypeError}), ('faast://:@/', {'instance': TypeError}), ('faast://%s' % ('a' * 32), {'instance': NotifyFaast, 'privacy_url': 'faast://a...a'}), ('faast://%s' % ('a' * 32), {'instance': NotifyFaast, 'include_image': False}), ('faast://%s' % ('a' * 32), {'instance': NotifyFaast, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('faast://%s' % ('a' * 32), {'instance': NotifyFaast, 'response': False, 'requests_response_code': 999}), ('faast://%s' % ('a' * 32), {'instance': NotifyFaast, 'test_requests_exceptions': True}))

def test_plugin_faast_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyFaast() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()