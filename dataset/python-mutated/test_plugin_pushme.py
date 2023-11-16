import requests
from apprise.plugins.NotifyPushMe import NotifyPushMe
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('pushme://', {'instance': TypeError}), ('pushme://:@/', {'instance': TypeError}), ('pushme://%s' % ('a' * 6), {'instance': NotifyPushMe, 'privacy_url': 'pushme://a...a/'}), ('pushme://?token=%s&status=yes' % ('b' * 6), {'instance': NotifyPushMe, 'privacy_url': 'pushme://b...b/'}), ('pushme://?token=%s&status=no' % ('b' * 6), {'instance': NotifyPushMe, 'privacy_url': 'pushme://b...b/'}), ('pushme://?token=%s&status=True' % ('b' * 6), {'instance': NotifyPushMe, 'privacy_url': 'pushme://b...b/'}), ('pushme://?push_key=%s&status=no' % ('p' * 6), {'instance': NotifyPushMe, 'privacy_url': 'pushme://p...p/'}), ('pushme://%s' % ('c' * 6), {'instance': NotifyPushMe, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pushme://%s' % ('d' * 7), {'instance': NotifyPushMe, 'response': False, 'requests_response_code': 999}), ('pushme://%s' % ('e' * 8), {'instance': NotifyPushMe, 'test_requests_exceptions': True}))

def test_plugin_pushme_urls():
    if False:
        print('Hello World!')
    '\n    NotifyPushMe() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()