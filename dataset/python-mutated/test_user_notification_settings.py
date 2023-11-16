from rest_framework import status
from sentry.models.notificationsetting import NotificationSetting
from sentry.models.notificationsettingoption import NotificationSettingOption
from sentry.models.notificationsettingprovider import NotificationSettingProvider
from sentry.notifications.types import NotificationSettingOptionValues, NotificationSettingTypes
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import control_silo_test
from sentry.types.integrations import ExternalProviders

class UserNotificationSettingsTestBase(APITestCase):
    endpoint = 'sentry-api-0-user-notification-settings'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.login_as(self.user)

@control_silo_test(stable=True)
class UserNotificationSettingsGetTest(UserNotificationSettingsTestBase):

    def test_simple(self):
        if False:
            while True:
                i = 10
        NotificationSetting.objects.update_settings(ExternalProviders.EMAIL, NotificationSettingTypes.ISSUE_ALERTS, NotificationSettingOptionValues.NEVER, user_id=self.user.id)
        NotificationSetting.objects.update_settings(ExternalProviders.EMAIL, NotificationSettingTypes.DEPLOY, NotificationSettingOptionValues.NEVER, user_id=self.user.id, organization=self.organization)
        NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.DEPLOY, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id, organization=self.organization)
        NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.WORKFLOW, NotificationSettingOptionValues.SUBSCRIBE_ONLY, user_id=self.user.id)
        response = self.get_success_response('me')
        assert response.data['alerts']['user'][self.user.id]['email'] == 'never'
        assert response.data['deploy']['organization'][self.organization.id]['email'] == 'never'
        assert response.data['deploy']['organization'][self.organization.id]['slack'] == 'always'
        assert response.data['workflow']['user'][self.user.id]['slack'] == 'subscribe_only'

    def test_notification_settings_empty(self):
        if False:
            print('Hello World!')
        response = self.get_success_response('me')
        assert response.data['preferences'] == {}

    def test_type_querystring(self):
        if False:
            i = 10
            return i + 15
        NotificationSetting.objects.update_settings(ExternalProviders.EMAIL, NotificationSettingTypes.ISSUE_ALERTS, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id, project=self.project)
        NotificationSetting.objects.update_settings(ExternalProviders.SLACK, NotificationSettingTypes.WORKFLOW, NotificationSettingOptionValues.ALWAYS, user_id=self.user.id, project=self.project)
        response = self.get_success_response('me', qs_params={'type': 'workflow'})
        assert 'alerts' not in response.data
        assert 'workflow' in response.data

    def test_invalid_querystring(self):
        if False:
            i = 10
            return i + 15
        self.get_error_response('me', qs_params={'type': 'invalid'}, status_code=status.HTTP_400_BAD_REQUEST)

    def test_invalid_user_id(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_error_response('invalid', status_code=status.HTTP_404_NOT_FOUND)

    def test_wrong_user_id(self):
        if False:
            for i in range(10):
                print('nop')
        other_user = self.create_user('bizbaz@example.com')
        self.get_error_response(other_user.id, status_code=status.HTTP_403_FORBIDDEN)

@control_silo_test(stable=True)
class UserNotificationSettingsUpdateTest(UserNotificationSettingsTestBase):
    method = 'put'

    def test_simple(self):
        if False:
            return 10
        assert NotificationSetting.objects.get_settings(provider=ExternalProviders.SLACK, type=NotificationSettingTypes.DEPLOY, user_id=self.user.id) == NotificationSettingOptionValues.DEFAULT
        self.get_success_response('me', deploy={'user': {'me': {'email': 'always', 'slack': 'always'}}}, status_code=status.HTTP_204_NO_CONTENT)
        assert NotificationSetting.objects.get_settings(provider=ExternalProviders.SLACK, type=NotificationSettingTypes.DEPLOY, user_id=self.user.id) == NotificationSettingOptionValues.ALWAYS

    def test_double_write(self):
        if False:
            while True:
                i = 10
        org = self.create_organization()
        self.create_member(user=self.user, organization=org)
        assert NotificationSetting.objects.get_settings(provider=ExternalProviders.SLACK, type=NotificationSettingTypes.DEPLOY, user_id=self.user.id) == NotificationSettingOptionValues.DEFAULT
        self.get_success_response('me', deploy={'user': {'me': {'email': 'always', 'slack': 'always'}}}, status_code=status.HTTP_204_NO_CONTENT)
        assert NotificationSetting.objects.get_settings(provider=ExternalProviders.SLACK, type=NotificationSettingTypes.DEPLOY, user_id=self.user.id) == NotificationSettingOptionValues.ALWAYS
        query_args = {'user_id': self.user.id, 'team_id': None, 'value': 'always', 'scope_type': 'user', 'scope_identifier': self.user.id, 'type': 'deploy'}
        assert NotificationSettingOption.objects.filter(**query_args).exists()
        assert NotificationSettingProvider.objects.filter(**query_args, provider='email')
        assert NotificationSettingProvider.objects.filter(**query_args, provider='slack')
        assert not NotificationSettingProvider.objects.filter(**query_args, provider='msteams')
        self.get_success_response('me', deploy={'user': {'me': {'email': 'default', 'slack': 'never'}}}, status_code=status.HTTP_204_NO_CONTENT)
        del query_args['value']
        assert not NotificationSettingProvider.objects.filter(**query_args, provider='email')
        assert NotificationSettingProvider.objects.filter(**query_args, value='never', provider='slack')

    def test_double_write_with_email_off(self):
        if False:
            print('Hello World!')
        org = self.create_organization()
        self.create_member(user=self.user, organization=org)
        project2 = self.create_project(organization=org)
        self.get_success_response('me', deploy={'user': {'me': {'email': 'never', 'slack': 'committed_only'}}, 'project': {project2.id: {'email': 'never', 'slack': 'always'}, self.project.id: {'email': 'never', 'slack': 'never'}}}, status_code=status.HTTP_204_NO_CONTENT)
        base_query_args = {'user_id': self.user.id, 'team_id': None, 'type': 'deploy'}
        query_args = {**base_query_args, 'scope_type': 'user', 'scope_identifier': self.user.id}
        assert NotificationSettingOption.objects.filter(**query_args, value='committed_only').exists()
        assert NotificationSettingProvider.objects.filter(**query_args, provider='email', value='never').exists()
        assert NotificationSettingProvider.objects.filter(**query_args, provider='slack', value='always').exists()
        assert not NotificationSettingProvider.objects.filter(**query_args, provider='msteams').exists()
        query_args = {**base_query_args, 'scope_type': 'project', 'scope_identifier': self.project.id}
        assert NotificationSettingOption.objects.filter(**query_args, value='never').exists()
        assert not NotificationSettingProvider.objects.filter(**query_args).exists()
        query_args = {**base_query_args, 'scope_type': 'project', 'scope_identifier': project2.id}
        assert NotificationSettingOption.objects.filter(**query_args, value='always').exists()
        assert not NotificationSettingProvider.objects.filter(**query_args).exists()

    def test_empty_payload(self):
        if False:
            i = 10
            return i + 15
        self.get_error_response('me', status_code=status.HTTP_400_BAD_REQUEST)

    def test_invalid_payload(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_error_response('me', invalid=1, status_code=status.HTTP_400_BAD_REQUEST)

    def test_malformed_payload(self):
        if False:
            return 10
        self.get_error_response('me', alerts=[1, 2], status_code=status.HTTP_400_BAD_REQUEST)

    def test_wrong_user_id(self):
        if False:
            print('Hello World!')
        user2 = self.create_user()
        self.get_error_response('me', deploy={'user': {user2.id: {'email': 'always', 'slack': 'always'}}}, status_code=status.HTTP_400_BAD_REQUEST)