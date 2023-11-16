import requests
from apprise.plugins.NotifyParsePlatform import NotifyParsePlatform
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('parsep://', {'instance': None}), ('parsep://:@/', {'instance': None}), ('parsep://%s' % ('a' * 32), {'instance': TypeError}), ('parsep://app_id@%s' % ('a' * 32), {'instance': TypeError}), ('parseps://:master_key@%s' % ('a' * 32), {'instance': TypeError}), ('parseps://localhost?app_id=%s&master_key=%s' % ('a' * 32, 'd' * 32), {'instance': NotifyParsePlatform, 'privacy_url': 'parseps://a...a:d...d@localhost'}), ('parsep://app_id:master_key@localhost:8080?device=ios', {'instance': NotifyParsePlatform}), ('parsep://app_id:master_key@localhost?device=invalid', {'instance': TypeError}), ('parseps://app_id:master_key@localhost', {'instance': NotifyParsePlatform}), ('parseps://app_id:master_key@localhost', {'instance': NotifyParsePlatform, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('parseps://app_id:master_key@localhost', {'instance': NotifyParsePlatform, 'response': False, 'requests_response_code': 999}), ('parseps://app_id:master_key@localhost', {'instance': NotifyParsePlatform, 'test_requests_exceptions': True}))

def test_plugin_parse_platform_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyParsePlatform() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()