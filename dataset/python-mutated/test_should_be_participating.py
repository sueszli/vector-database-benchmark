from sentry.models.groupsubscription import GroupSubscription
from sentry.notifications.helpers import should_be_participating, where_should_be_participating
from sentry.notifications.types import NotificationScopeType, NotificationSettingOptionValues
from sentry.services.hybrid_cloud.actor import RpcActor
from sentry.silo import SiloMode
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import assume_test_silo_mode, control_silo_test
from sentry.types.integrations import ExternalProviders

class ShouldBeParticipatingTest(TestCase):

    def test_subscription_on_notification_settings_always(self):
        if False:
            while True:
                i = 10
        subscription = GroupSubscription(is_active=True)
        value = should_be_participating(subscription, NotificationSettingOptionValues.ALWAYS)
        assert value

    def test_subscription_off_notification_settings_always(self):
        if False:
            for i in range(10):
                print('nop')
        subscription = GroupSubscription(is_active=False)
        value = should_be_participating(subscription, NotificationSettingOptionValues.ALWAYS)
        assert not value

    def test_subscription_null_notification_settings_always(self):
        if False:
            while True:
                i = 10
        value = should_be_participating(None, NotificationSettingOptionValues.ALWAYS)
        assert value

    def test_subscription_on_notification_setting_never(self):
        if False:
            while True:
                i = 10
        subscription = GroupSubscription(is_active=True)
        value = should_be_participating(subscription, NotificationSettingOptionValues.NEVER)
        assert not value

    def test_subscription_off_notification_setting_never(self):
        if False:
            while True:
                i = 10
        subscription = GroupSubscription(is_active=False)
        value = should_be_participating(subscription, NotificationSettingOptionValues.NEVER)
        assert not value

    def test_subscription_on_subscribe_only(self):
        if False:
            i = 10
            return i + 15
        subscription = GroupSubscription(is_active=True)
        value = should_be_participating(subscription, NotificationSettingOptionValues.SUBSCRIBE_ONLY)
        assert value

    def test_subscription_off_subscribe_only(self):
        if False:
            return 10
        subscription = GroupSubscription(is_active=False)
        value = should_be_participating(subscription, NotificationSettingOptionValues.SUBSCRIBE_ONLY)
        assert not value

    def test_subscription_null_subscribe_only(self):
        if False:
            i = 10
            return i + 15
        value = should_be_participating(None, NotificationSettingOptionValues.SUBSCRIBE_ONLY)
        assert not value

@control_silo_test(stable=True)
class WhereShouldBeParticipatingTest(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with assume_test_silo_mode(SiloMode.REGION):
            self.user = RpcActor.from_orm_user(self.create_user())

    def test_where_should_be_participating(self):
        if False:
            return 10
        subscription = GroupSubscription(is_active=True)
        notification_settings = {self.user: {NotificationScopeType.USER: {ExternalProviders.EMAIL: NotificationSettingOptionValues.ALWAYS, ExternalProviders.SLACK: NotificationSettingOptionValues.SUBSCRIBE_ONLY, ExternalProviders.PAGERDUTY: NotificationSettingOptionValues.NEVER}}}
        providers = where_should_be_participating(self.user, subscription, notification_settings)
        assert providers == [ExternalProviders.EMAIL, ExternalProviders.SLACK]

    def test_subscription_null(self):
        if False:
            print('Hello World!')
        notification_settings = {self.user: {NotificationScopeType.USER: {ExternalProviders.EMAIL: NotificationSettingOptionValues.ALWAYS, ExternalProviders.SLACK: NotificationSettingOptionValues.SUBSCRIBE_ONLY, ExternalProviders.PAGERDUTY: NotificationSettingOptionValues.NEVER}}}
        providers = where_should_be_participating(self.user, None, notification_settings)
        assert providers == [ExternalProviders.EMAIL]