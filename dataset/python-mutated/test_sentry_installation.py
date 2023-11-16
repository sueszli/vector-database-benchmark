from unittest.mock import patch
from jwt import ExpiredSignatureError
from sentry.integrations.jira.views import UNABLE_TO_VERIFY_INSTALLATION
from sentry.integrations.utils import AtlassianConnectValidationError
from sentry.models.integrations.integration import Integration
from sentry.testutils.cases import APITestCase
from sentry.utils.http import absolute_uri
REFRESH_REQUIRED = b'This page has expired, please refresh to configure your Sentry integration'
CLICK_TO_FINISH = b'Finish Installation in Sentry'

class JiraSentryInstallationViewTestCase(APITestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.path = absolute_uri('extensions/jira/ui-hook/') + '?xdm_e=base_url'
        self.user.name = 'Sentry Admin'
        self.user.save()
        self.integration = Integration.objects.create(provider='jira', name='Example Jira')

class JiraSentryInstallationViewErrorsTest(JiraSentryInstallationViewTestCase):

    @patch('sentry.integrations.jira.views.sentry_installation.get_integration_from_request', side_effect=ExpiredSignatureError())
    def test_expired_signature_error(self, mock_get_integration_from_request):
        if False:
            return 10
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert REFRESH_REQUIRED in response.content

    @patch('sentry.integrations.jira.views.sentry_installation.get_integration_from_request', side_effect=AtlassianConnectValidationError())
    def test_expired_invalid_installation_error(self, mock_get_integration_from_request):
        if False:
            i = 10
            return i + 15
        response = self.client.get(self.path)
        assert response.status_code == 200
        assert UNABLE_TO_VERIFY_INSTALLATION.encode() in response.content

class JiraSentryInstallationViewTest(JiraSentryInstallationViewTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.login_as(self.user)

    def assert_no_errors(self, response):
        if False:
            print('Hello World!')
        assert REFRESH_REQUIRED not in response.content
        assert UNABLE_TO_VERIFY_INSTALLATION.encode() not in response.content

    @patch('sentry.integrations.jira.views.sentry_installation.get_integration_from_request')
    def test_simple_get(self, mock_get_integration_from_request):
        if False:
            for i in range(10):
                print('nop')
        mock_get_integration_from_request.return_value = self.integration
        response = self.client.get(self.path)
        assert response.status_code == 200
        self.assert_no_errors(response)
        assert CLICK_TO_FINISH in response.content