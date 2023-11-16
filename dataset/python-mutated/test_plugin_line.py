from apprise.plugins.NotifyLine import NotifyLine
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('line://', {'instance': TypeError}), ('line://%20/', {'instance': TypeError}), ('line://token', {'instance': NotifyLine, 'notify_response': False}), ('line://token=/target', {'instance': NotifyLine, 'privacy_url': 'line://****/t...t?'}), ('line://token/target?image=no', {'instance': NotifyLine}), ('line://a/very/long/token=/target?image=no', {'instance': NotifyLine}), ('line://?token=token&to=target1', {'instance': NotifyLine, 'privacy_url': 'line://****/t...1?'}), ('line://token/target', {'instance': NotifyLine, 'response': False, 'requests_response_code': 999}), ('line://token/target', {'instance': NotifyLine, 'test_requests_exceptions': True}))

def test_plugin_line_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyLine() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()