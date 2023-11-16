from apprise.plugins.NotifyPushy import NotifyPushy
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
GOOD_RESPONSE = {'success': True}
apprise_url_tests = (('pushy://', {'instance': TypeError}), ('pushy://:@/', {'instance': TypeError}), ('pushy://apikey', {'instance': NotifyPushy, 'notify_response': False, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/topic', {'instance': NotifyPushy, 'notify_response': False, 'requests_response_text': {'success': False}}), ('pushy://apikey/topic', {'instance': NotifyPushy, 'notify_response': False, 'requests_response_text': '}'}), ('pushy://apikey/%20(', {'instance': NotifyPushy, 'notify_response': False, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/@device', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE, 'privacy_url': 'pushy://a...y/@device/'}), ('pushy://apikey/topic', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE, 'privacy_url': 'pushy://a...y/#topic/'}), ('pushy://apikey/device/?sound=alarm.aiff', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/device/?badge=100', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/device/?badge=invalid', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/device/?badge=-12', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://_/@device/#topic?key=apikey', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://apikey/?to=@device', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE}), ('pushy://_/@device/#topic?key=apikey', {'instance': NotifyPushy, 'response': False, 'requests_response_code': 999, 'requests_response_text': GOOD_RESPONSE, 'privacy_url': 'pushy://a...y/#topic/@device/'}), ('pushy://_/@device/#topic?key=apikey', {'instance': NotifyPushy, 'requests_response_text': GOOD_RESPONSE, 'test_requests_exceptions': True}))

def test_plugin_pushy_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyPushy() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()