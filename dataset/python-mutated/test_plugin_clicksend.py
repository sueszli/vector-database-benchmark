from apprise.plugins.NotifyClickSend import NotifyClickSend
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('clicksend://', {'instance': TypeError}), ('clicksend://:@/', {'instance': TypeError}), ('clicksend://user:pass@{}/{}/{}'.format('1' * 9, '2' * 15, 'a' * 13), {'instance': NotifyClickSend, 'notify_response': False}), ('clicksend://user:pass@{}?batch=yes'.format('3' * 14), {'instance': NotifyClickSend}), ('clicksend://user:pass@{}?batch=yes&to={}'.format('3' * 14, '6' * 14), {'instance': NotifyClickSend, 'privacy_url': 'clicksend://user:****'}), ('clicksend://user:pass@{}?batch=no'.format('3' * 14), {'instance': NotifyClickSend}), ('clicksend://user:pass@{}'.format('3' * 14), {'instance': NotifyClickSend, 'response': False, 'requests_response_code': 999}), ('clicksend://user:pass@{}'.format('3' * 14), {'instance': NotifyClickSend, 'test_requests_exceptions': True}))

def test_plugin_clicksend_urls():
    if False:
        print('Hello World!')
    '\n    NotifyClickSend() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()