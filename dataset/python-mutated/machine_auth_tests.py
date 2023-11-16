from unittest.mock import call, Mock, patch
from superset.extensions import machine_auth_provider_factory
from tests.integration_tests.base_tests import SupersetTestCase

class MachineAuthProviderTests(SupersetTestCase):

    def test_get_auth_cookies(self):
        if False:
            print('Hello World!')
        user = self.get_user('admin')
        auth_cookies = machine_auth_provider_factory.instance.get_auth_cookies(user)
        self.assertIsNotNone(auth_cookies['session'])

    @patch('superset.utils.machine_auth.MachineAuthProvider.get_auth_cookies')
    def test_auth_driver_user(self, get_auth_cookies):
        if False:
            i = 10
            return i + 15
        user = self.get_user('admin')
        driver = Mock()
        get_auth_cookies.return_value = {'session': 'session_val', 'other_cookie': 'other_val'}
        machine_auth_provider_factory.instance.authenticate_webdriver(driver, user)
        driver.add_cookie.assert_has_calls([call({'name': 'session', 'value': 'session_val'}), call({'name': 'other_cookie', 'value': 'other_val'})])

    @patch('superset.utils.machine_auth.request')
    def test_auth_driver_request(self, request):
        if False:
            print('Hello World!')
        driver = Mock()
        request.cookies = {'session': 'session_val', 'other_cookie': 'other_val'}
        machine_auth_provider_factory.instance.authenticate_webdriver(driver, None)
        driver.add_cookie.assert_has_calls([call({'name': 'session', 'value': 'session_val'}), call({'name': 'other_cookie', 'value': 'other_val'})])