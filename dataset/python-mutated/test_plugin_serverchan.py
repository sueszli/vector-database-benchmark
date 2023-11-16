from apprise.plugins.NotifyServerChan import NotifyServerChan
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('schan://', {'instance': TypeError}), ('schan://a_bd_/', {'instance': TypeError}), ('schan://12345678', {'instance': NotifyServerChan, 'privacy_url': 'schan://1...8'}), ('schan://{}'.format('a' * 8), {'instance': NotifyServerChan, 'response': False, 'requests_response_code': 999}), ('schan://{}'.format('a' * 8), {'instance': NotifyServerChan, 'test_requests_exceptions': True}))

def test_plugin_serverchan_urls():
    if False:
        print('Hello World!')
    '\n    NotifyServerChan() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()