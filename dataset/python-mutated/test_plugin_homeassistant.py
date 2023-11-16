from unittest import mock
import requests
from apprise import Apprise
from apprise.plugins.NotifyHomeAssistant import NotifyHomeAssistant
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('hassio://:@/', {'instance': TypeError}), ('hassio://', {'instance': TypeError}), ('hassios://', {'instance': TypeError}), ('hassio://user@localhost', {'instance': TypeError}), ('hassio://localhost/long-lived-access-token', {'instance': NotifyHomeAssistant}), ('hassio://user:pass@localhost/long-lived-access-token/', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassio://user:****@localhost/l...n'}), ('hassio://localhost:80/long-lived-access-token', {'instance': NotifyHomeAssistant}), ('hassio://user@localhost:8123/llat', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassio://user@localhost/l...t'}), ('hassios://localhost/llat?nid=!%', {'instance': TypeError}), ('hassios://localhost/llat?nid=abcd', {'instance': NotifyHomeAssistant}), ('hassios://user:pass@localhost/llat', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassios://user:****@localhost/l...t'}), ('hassios://localhost:8443/path/llat/', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassios://localhost:8443/path/l...t'}), ('hassio://localhost:8123/a/path?accesstoken=llat', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassio://localhost/a/path/l...t'}), ('hassios://user:password@localhost:80/llat/', {'instance': NotifyHomeAssistant, 'privacy_url': 'hassios://user:****@localhost:80'}), ('hassio://user:pass@localhost:8123/llat', {'instance': NotifyHomeAssistant, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('hassio://user:pass@localhost/llat', {'instance': NotifyHomeAssistant, 'response': False, 'requests_response_code': 999}), ('hassio://user:pass@localhost/llat', {'instance': NotifyHomeAssistant, 'test_requests_exceptions': True}))

def test_plugin_homeassistant_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyHomeAssistant() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_homeassistant_general(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyHomeAssistant() General Checks\n\n    '
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('hassio://localhost/accesstoken')
    assert isinstance(obj, NotifyHomeAssistant) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'http://localhost:8123/api/services/persistent_notification/create'