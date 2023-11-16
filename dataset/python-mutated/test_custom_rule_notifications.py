import time
from datetime import datetime, timedelta, timezone
from unittest import mock
from sentry.dynamic_sampling.tasks.custom_rule_notifications import MIN_SAMPLES_FOR_NOTIFICATION, clean_custom_rule_notifications, custom_rule_notifications, get_num_samples
from sentry.models.dynamicsampling import CustomDynamicSamplingRule
from sentry.testutils.cases import SnubaTestCase, TestCase
from sentry.utils.samples import load_data

class CustomRuleNotificationsTest(TestCase, SnubaTestCase):

    def create_transaction(self):
        if False:
            print('Hello World!')
        data = load_data('transaction')
        return self.store_event(data, project_id=self.project.id)

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.user = self.create_user(email='radu@sentry.io', username='raduw', name='RaduW')
        now = datetime.now(timezone.utc) - timedelta(minutes=2)
        condition = {'op': 'and', 'inner': [{'op': 'eq', 'name': 'event.environment', 'value': 'dev'}, {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}]}
        query = 'event.type:transaction environment:dev'
        self.rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=now, end=now + timedelta(days=1), project_ids=[], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query=query, created_by_id=self.user.id)

    def test_get_num_samples(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that the num_samples function returns the correct number of samples\n        '
        num_samples = get_num_samples(self.rule)
        assert num_samples == 0
        self.create_transaction()
        self.create_transaction()
        self.create_transaction()
        num_samples = get_num_samples(self.rule)
        assert num_samples == 3

    @mock.patch('sentry.dynamic_sampling.tasks.custom_rule_notifications.send_notification')
    def test_email_is_sent_when_enough_samples_have_been_collected(self, send_notification_mock):
        if False:
            return 10
        for idx in range(MIN_SAMPLES_FOR_NOTIFICATION):
            self.create_transaction()
        time.sleep(1.0)
        self.rule.refresh_from_db()
        assert not self.rule.notification_sent
        with self.tasks():
            custom_rule_notifications()
        send_notification_mock.assert_called_once()
        self.rule.refresh_from_db()
        assert self.rule.notification_sent

    def test_clean_custom_rule_notifications(self):
        if False:
            print('Hello World!')
        '\n        Tests that expired rules are deactivated\n        '
        start = datetime.now(timezone.utc) - timedelta(hours=2)
        end = datetime.now(timezone.utc) - timedelta(minutes=2)
        condition = {'op': 'eq', 'name': 'event.tags.event.type', 'value': 'transaction'}
        query = 'event.type:transaction'
        expired_rule = CustomDynamicSamplingRule.update_or_create(condition=condition, start=start, end=end, project_ids=[], organization_id=self.organization.id, num_samples=100, sample_rate=0.5, query=query, created_by_id=self.user.id)
        assert expired_rule.is_active
        assert self.rule.is_active
        with self.tasks():
            clean_custom_rule_notifications()
        self.rule.refresh_from_db()
        assert self.rule.is_active
        expired_rule.refresh_from_db()
        assert not expired_rule.is_active