from sentry.models.user import User
from sentry.notifications.helpers import get_highest_notification_setting_value, get_most_specific_notification_setting_value
from sentry.notifications.types import NotificationScopeType, NotificationSettingOptionValues, NotificationSettingTypes
from sentry.services.hybrid_cloud.actor import ActorType, RpcActor
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import control_silo_test
from sentry.types.integrations import ExternalProviders

@control_silo_test(stable=True)
class GetMostSpecificNotificationSettingValueTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.user = self.create_user()

    def test_get_most_specific_notification_setting_value_empty_workflow(self):
        if False:
            for i in range(10):
                print('nop')
        value = get_most_specific_notification_setting_value(notification_settings_by_scope={}, recipient=RpcActor(id=self.user.id, actor_type=ActorType.USER), parent_id=1, type=NotificationSettingTypes.WORKFLOW)
        assert value == NotificationSettingOptionValues.SUBSCRIBE_ONLY

    def test_get_most_specific_notification_setting_value_empty_alerts(self):
        if False:
            i = 10
            return i + 15
        value = get_most_specific_notification_setting_value(notification_settings_by_scope={}, recipient=RpcActor(id=self.user.id, actor_type=ActorType.USER), parent_id=1, type=NotificationSettingTypes.ISSUE_ALERTS)
        assert value == NotificationSettingOptionValues.ALWAYS

    def test_get_most_specific_notification_setting_value_user(self):
        if False:
            i = 10
            return i + 15
        notification_settings_by_scope = {NotificationScopeType.USER: {self.user.id: {ExternalProviders.SLACK: NotificationSettingOptionValues.NEVER, ExternalProviders.EMAIL: NotificationSettingOptionValues.ALWAYS}}}
        value = get_most_specific_notification_setting_value(notification_settings_by_scope, recipient=RpcActor(id=self.user.id, actor_type=ActorType.USER), parent_id=1, type=NotificationSettingTypes.ISSUE_ALERTS)
        assert value == NotificationSettingOptionValues.ALWAYS

    def test_get_most_specific_notification_setting_value(self):
        if False:
            while True:
                i = 10
        project_id = 1
        notification_settings_by_scope = {NotificationScopeType.USER: {self.user.id: {ExternalProviders.SLACK: NotificationSettingOptionValues.NEVER, ExternalProviders.EMAIL: NotificationSettingOptionValues.ALWAYS}}, NotificationScopeType.PROJECT: {project_id: {ExternalProviders.SLACK: NotificationSettingOptionValues.NEVER, ExternalProviders.EMAIL: NotificationSettingOptionValues.NEVER}}}
        value = get_most_specific_notification_setting_value(notification_settings_by_scope, recipient=RpcActor(id=self.user.id, actor_type=ActorType.USER), parent_id=project_id, type=NotificationSettingTypes.ISSUE_ALERTS)
        assert value == NotificationSettingOptionValues.NEVER

class GetHighestNotificationSettingValueTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.user = User(id=1)

    def test_get_highest_notification_setting_value_empty(self):
        if False:
            for i in range(10):
                print('nop')
        assert get_highest_notification_setting_value({}) is None

    def test_get_highest_notification_setting_value(self):
        if False:
            return 10
        value = get_highest_notification_setting_value({ExternalProviders.SLACK: NotificationSettingOptionValues.NEVER, ExternalProviders.EMAIL: NotificationSettingOptionValues.ALWAYS})
        assert value == NotificationSettingOptionValues.ALWAYS

    def test_get_highest_notification_setting_value_never(self):
        if False:
            while True:
                i = 10
        value = get_highest_notification_setting_value({ExternalProviders.SLACK: NotificationSettingOptionValues.NEVER, ExternalProviders.EMAIL: NotificationSettingOptionValues.NEVER})
        assert value == NotificationSettingOptionValues.NEVER