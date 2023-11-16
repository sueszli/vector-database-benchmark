from rest_framework import status
from sentry.models.notificationsettingprovider import NotificationSettingProvider
from sentry.notifications.types import NotificationScopeEnum, NotificationSettingEnum, NotificationSettingsOptionEnum
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test
from sentry.types.integrations import ExternalProviderEnum

class UserNotificationSettingsProvidersBaseTest(APITestCase):
    endpoint = 'sentry-api-0-user-notification-providers'

@control_silo_test(stable=True)
class UserNotificationSettingsProvidersGetTest(UserNotificationSettingsProvidersBaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        other_user = self.create_user()
        NotificationSettingProvider.objects.create(user_id=self.user.id, scope_type=NotificationScopeEnum.ORGANIZATION.value, scope_identifier=self.organization.id, type=NotificationSettingEnum.ISSUE_ALERTS.value, provider=ExternalProviderEnum.SLACK.value, value=NotificationSettingsOptionEnum.ALWAYS.value)
        NotificationSettingProvider.objects.create(user_id=self.user.id, scope_type=NotificationScopeEnum.ORGANIZATION.value, scope_identifier=self.organization.id, type=NotificationSettingEnum.ISSUE_ALERTS.value, provider=ExternalProviderEnum.EMAIL.value, value=NotificationSettingsOptionEnum.ALWAYS.value)
        NotificationSettingProvider.objects.create(user_id=self.user.id, scope_type=NotificationScopeEnum.ORGANIZATION.value, scope_identifier=self.organization.id, type=NotificationSettingEnum.WORKFLOW.value, provider=ExternalProviderEnum.EMAIL.value, value=NotificationSettingsOptionEnum.ALWAYS.value)
        NotificationSettingProvider.objects.create(user_id=other_user.id, scope_type=NotificationScopeEnum.ORGANIZATION.value, scope_identifier=self.organization.id, type=NotificationSettingEnum.ISSUE_ALERTS.value, provider=ExternalProviderEnum.SLACK.value, value=NotificationSettingsOptionEnum.ALWAYS.value)
        response = self.get_success_response('me', type='alerts').data
        assert len(response) == 2
        slack_item = next((item for item in response if item['provider'] == 'slack'))
        email_item = next((item for item in response if item['provider'] == 'email'))
        assert slack_item['scopeType'] == 'organization'
        assert slack_item['scopeIdentifier'] == str(self.organization.id)
        assert slack_item['user_id'] == str(self.user.id)
        assert slack_item['team_id'] is None
        assert slack_item['value'] == 'always'
        assert slack_item['type'] == 'alerts'
        assert slack_item['provider'] == 'slack'
        assert email_item['provider'] == 'email'
        response = self.get_success_response('me').data
        assert len(response) == 3
        alert_slack_item = next((item for item in response if item['provider'] == 'slack' and item['type'] == 'alerts'))
        assert alert_slack_item['value'] == 'always'
        workflow_email_item = next((item for item in response if item['provider'] == 'email' and item['type'] == 'workflow'))
        assert workflow_email_item['value'] == 'always'
        alert_slack_item = next((item for item in response if item['provider'] == 'email' and item['type'] == 'alerts'))
        assert alert_slack_item['value'] == 'always'

    def test_invalid_type(self):
        if False:
            return 10
        response = self.get_error_response('me', type='invalid', status_code=status.HTTP_400_BAD_REQUEST)
        assert response.data['type'] == ['Invalid type']

@control_silo_test(stable=True)
class UserNotificationSettingsProvidersPutTest(UserNotificationSettingsProvidersBaseTest):
    method = 'PUT'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(self.user)

    def test_simple(self):
        if False:
            return 10
        response = self.get_success_response('me', user_id=self.user.id, scope_type='organization', scope_identifier=self.organization.id, type='alerts', status_code=status.HTTP_201_CREATED, value='always', providers=['slack'])
        assert NotificationSettingProvider.objects.filter(user_id=self.user.id, scope_type=NotificationScopeEnum.ORGANIZATION.value, scope_identifier=self.organization.id, type=NotificationSettingEnum.ISSUE_ALERTS.value, value=NotificationSettingsOptionEnum.ALWAYS.value, provider=ExternalProviderEnum.SLACK.value).exists()
        assert len(response.data) == 3

    def test_invalid_scope_type(self):
        if False:
            i = 10
            return i + 15
        response = self.get_error_response('me', user_id=self.user.id, scope_type='project', scope_identifier=self.project.id, type='alerts', status_code=status.HTTP_400_BAD_REQUEST, providers=['slack'])
        assert response.data['scopeType'] == ['Invalid scope type']

    def test_invalid_provider(self):
        if False:
            print('Hello World!')
        response = self.get_error_response('me', user_id=self.user.id, scope_type='organization', scope_identifier=self.organization.id, type='alerts', status_code=status.HTTP_400_BAD_REQUEST, providers=['github'])
        assert response.data['providers'] == ['Invalid provider']