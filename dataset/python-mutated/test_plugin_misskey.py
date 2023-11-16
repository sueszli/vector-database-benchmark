import os
from apprise.plugins.NotifyMisskey import NotifyMisskey
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('misskey://', {'instance': None}), ('misskey://:@/', {'instance': None}), ('misskey://hostname', {'instance': TypeError}), ('misskey://access_token@hostname', {'instance': NotifyMisskey}), ('misskeys://access_token@hostname', {'instance': NotifyMisskey, 'privacy_url': 'misskeys://a...n@hostname/'}), ('misskey://hostname/?token=abcd123', {'instance': NotifyMisskey, 'privacy_url': 'misskey://a...3@hostname'}), ('misskeys://access_token@hostname:8443', {'instance': NotifyMisskey}), ('misskey://access_token@hostname?visibility=invalid', {'instance': TypeError}), ('misskeys://access_token@hostname?visibility=private', {'instance': NotifyMisskey}), ('misskeys://access_token@hostname', {'instance': NotifyMisskey, 'response': False, 'requests_response_code': 999}), ('misskeys://access_token@hostname', {'instance': NotifyMisskey, 'test_requests_exceptions': True}))

def test_plugin_misskey_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMisskey() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()