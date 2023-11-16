import requests
from apprise.plugins.NotifySpontit import NotifySpontit
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('spontit://', {'instance': TypeError}), ('spontit://:@/', {'instance': TypeError}), ('spontit://%s' % ('a' * 100), {'instance': TypeError}), ('spontit://user@%%20_', {'instance': TypeError}), ('spontit://%s@%s' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit, 'privacy_url': 'spontit://{}@b...b/'.format('u' * 11)}), ('spontit://%s@%s/#!!' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://%s@%s/#abcd' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://%s@%s/?subtitle=Test' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://%s@%s/?subtitle=%s' % ('u' * 11, 'b' * 100, 'c' * 300), {'instance': NotifySpontit}), ('spontit://{}@{}/#1245%2Fabcd'.format('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://{}@{}/#1245%2Fabcd/defg'.format('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://{}@{}/?to=#1245/abcd'.format('u' * 11, 'b' * 100), {'instance': NotifySpontit}), ('spontit://%s@%s' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('spontit://%s@%s' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit, 'response': False, 'requests_response_code': 999}), ('spontit://%s@%s' % ('u' * 11, 'b' * 100), {'instance': NotifySpontit, 'test_requests_exceptions': True}))

def test_plugin_spontit_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySpontit() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()