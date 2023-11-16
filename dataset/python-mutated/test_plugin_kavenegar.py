from apprise.plugins.NotifyKavenegar import NotifyKavenegar
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('kavenegar://', {'instance': TypeError}), ('kavenegar://:@/', {'instance': TypeError}), ('kavenegar://{}/{}/{}'.format('1' * 10, '2' * 15, 'a' * 13), {'instance': NotifyKavenegar, 'notify_response': False}), ('kavenegar://{}/{}'.format('a' * 24, '3' * 14), {'instance': NotifyKavenegar, 'privacy_url': 'kavenegar://a...a/'}), ('kavenegar://{}?to={}'.format('a' * 24, '3' * 14), {'instance': NotifyKavenegar, 'privacy_url': 'kavenegar://a...a/'}), ('kavenegar://{}@{}/{}'.format('1' * 14, 'b' * 24, '3' * 14), {'instance': NotifyKavenegar, 'privacy_url': 'kavenegar://{}@b...b/'.format('1' * 14)}), ('kavenegar://{}@{}/{}'.format('a' * 14, 'b' * 24, '3' * 14), {'instance': TypeError}), ('kavenegar://{}@{}/{}'.format('3' * 4, 'b' * 24, '3' * 14), {'instance': TypeError}), ('kavenegar://{}/{}?from={}'.format('b' * 24, '3' * 14, '1' * 14), {'instance': NotifyKavenegar, 'privacy_url': 'kavenegar://{}@b...b/'.format('1' * 14)}), ('kavenegar://{}/{}'.format('b' * 24, '4' * 14), {'instance': NotifyKavenegar, 'response': False, 'requests_response_code': 999}), ('kavenegar://{}/{}'.format('c' * 24, '5' * 14), {'instance': NotifyKavenegar, 'test_requests_exceptions': True}))

def test_plugin_kavenegar_urls():
    if False:
        print('Hello World!')
    '\n    NotifyKavenegar() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()