from datetime import datetime, timedelta
from unittest.mock import patch
import jwt
import pytest
from zoneinfo import ZoneInfo
from django.conf import settings
from django.utils import timezone
from freezegun import freeze_time
from posthog.jwt import PosthogJwtAudience
from posthog.models.insight import Insight
from posthog.models.subscription import UNSUBSCRIBE_TOKEN_EXP_DAYS, Subscription, get_unsubscribe_token, unsubscribe_using_token
from posthog.test.base import BaseTest

@patch.object(settings, 'SECRET_KEY', 'not-so-secret')
@freeze_time('2022-01-01')
class TestSubscription(BaseTest):

    def _create_insight_subscription(self, **kwargs):
        if False:
            i = 10
            return i + 15
        insight = Insight.objects.create(team=self.team)
        params = dict(team=self.team, title='My Subscription', insight=insight, target_type='email', target_value='tests@posthog.com', frequency='weekly', interval=2, start_date=datetime(2022, 1, 1, 0, 0, 0, 0).replace(tzinfo=ZoneInfo('UTC')))
        params.update(**kwargs)
        return Subscription.objects.create(**params)

    def test_creation(self):
        if False:
            i = 10
            return i + 15
        subscription = self._create_insight_subscription()
        subscription.save()
        assert subscription.title == 'My Subscription'
        subscription.set_next_delivery_date(datetime(2022, 1, 2, 0, 0, 0).replace(tzinfo=ZoneInfo('UTC')))
        assert subscription.next_delivery_date == datetime(2022, 1, 15, 0, 0).replace(tzinfo=ZoneInfo('UTC'))

    def test_update_next_delivery_date_on_save(self):
        if False:
            for i in range(10):
                print('nop')
        subscription = self._create_insight_subscription()
        subscription.save()
        assert subscription.next_delivery_date >= timezone.now()

    def test_only_updates_next_delivery_date_if_rrule_changes(self):
        if False:
            for i in range(10):
                print('nop')
        subscription = self._create_insight_subscription()
        subscription.save()
        assert subscription.next_delivery_date
        old_date = subscription.next_delivery_date
        subscription.start_date = datetime(2023, 1, 1, 0, 0, 0, 0).replace(tzinfo=ZoneInfo('UTC'))
        subscription.save()
        assert old_date != subscription.next_delivery_date
        old_date = subscription.next_delivery_date
        subscription.title = 'My new title'
        subscription.target_value = 'other@example.com'
        subscription.save()
        assert old_date == subscription.next_delivery_date

    def test_generating_token(self):
        if False:
            print('Hello World!')
        subscription = self._create_insight_subscription(target_value='test1@posthog.com,test2@posthog.com,test3@posthog.com')
        subscription.save()
        token = get_unsubscribe_token(subscription, 'test2@posthog.com')
        assert token.startswith('ey')
        info = jwt.decode(token, 'not-so-secret', audience=PosthogJwtAudience.UNSUBSCRIBE.value, algorithms=['HS256'])
        assert info['id'] == subscription.id
        assert info['email'] == 'test2@posthog.com'
        assert info['exp'] == 1643587200

    def test_unsubscribe_using_token_succeeds(self):
        if False:
            print('Hello World!')
        subscription = self._create_insight_subscription(target_value='test1@posthog.com,test2@posthog.com,test3@posthog.com')
        subscription.save()
        token = get_unsubscribe_token(subscription, 'test2@posthog.com')
        subscription = unsubscribe_using_token(token)
        assert subscription.target_value == 'test1@posthog.com,test3@posthog.com'

    def test_unsubscribe_using_token_fails_if_too_old(self):
        if False:
            print('Hello World!')
        subscription = self._create_insight_subscription(target_value='test1@posthog.com,test2@posthog.com,test3@posthog.com')
        subscription.save()
        token = get_unsubscribe_token(subscription, 'test2@posthog.com')
        with freeze_time(datetime(2022, 1, 1) + timedelta(days=UNSUBSCRIBE_TOKEN_EXP_DAYS + 1)):
            with pytest.raises(jwt.exceptions.ExpiredSignatureError):
                unsubscribe_using_token(token)
        with freeze_time(datetime(2022, 1, 1) + timedelta(days=UNSUBSCRIBE_TOKEN_EXP_DAYS - 1)):
            subscription = unsubscribe_using_token(token)
            assert 'test2@posthog.com' not in subscription.target_value

    def test_unsubscribe_does_nothing_if_already_unsubscribed(self):
        if False:
            while True:
                i = 10
        subscription = self._create_insight_subscription(target_value='test1@posthog.com,test3@posthog.com')
        subscription.save()
        token = get_unsubscribe_token(subscription, 'test2@posthog.com')
        assert subscription.target_value == 'test1@posthog.com,test3@posthog.com'
        subscription = unsubscribe_using_token(token)
        assert subscription.target_value == 'test1@posthog.com,test3@posthog.com'

    def test_unsubscribe_deletes_subscription_if_last_subscriber(self):
        if False:
            return 10
        subscription = self._create_insight_subscription(target_value='test1@posthog.com,test2@posthog.com')
        subscription.save()
        assert not subscription.deleted
        token = get_unsubscribe_token(subscription, 'test1@posthog.com')
        subscription = unsubscribe_using_token(token)
        assert not subscription.deleted
        token = get_unsubscribe_token(subscription, 'test2@posthog.com')
        subscription = unsubscribe_using_token(token)
        assert subscription.deleted

    def test_complex_rrule_configuration(self):
        if False:
            for i in range(10):
                print('nop')
        subscription = self._create_insight_subscription(interval=2, frequency='monthly', bysetpos=-1, byweekday=['wednesday', 'friday'])
        subscription.save()
        assert subscription.next_delivery_date == datetime(2022, 1, 28, 0, 0).replace(tzinfo=ZoneInfo('UTC'))
        subscription.set_next_delivery_date(subscription.next_delivery_date)
        assert subscription.next_delivery_date == datetime(2022, 3, 30, 0, 0).replace(tzinfo=ZoneInfo('UTC'))
        subscription.set_next_delivery_date(subscription.next_delivery_date)
        assert subscription.next_delivery_date == datetime(2022, 5, 27, 0, 0).replace(tzinfo=ZoneInfo('UTC'))

    def test_should_work_for_nth_days(self):
        if False:
            for i in range(10):
                print('nop')
        subscription = self._create_insight_subscription(interval=1, frequency='monthly', bysetpos=3, byweekday=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
        subscription.save()
        assert subscription.next_delivery_date == datetime(2022, 1, 3, 0, 0).replace(tzinfo=ZoneInfo('UTC'))
        subscription.set_next_delivery_date(subscription.next_delivery_date)
        assert subscription.next_delivery_date == datetime(2022, 2, 3, 0, 0).replace(tzinfo=ZoneInfo('UTC'))

    def test_should_ignore_bysetpos_if_missing_weeekday(self):
        if False:
            return 10
        subscription = self._create_insight_subscription(interval=1, frequency='monthly', bysetpos=3)
        subscription.save()
        assert subscription.next_delivery_date == datetime(2022, 2, 1, 0, 0).replace(tzinfo=ZoneInfo('UTC'))

    def test_subscription_summary(self):
        if False:
            while True:
                i = 10
        subscription = self._create_insight_subscription(interval=1, frequency='monthly', bysetpos=None)
        assert subscription.summary == 'sent every month'
        subscription = self._create_insight_subscription(interval=2, frequency='monthly', byweekday=['wednesday'], bysetpos=1)
        assert subscription.summary == 'sent every 2 months on the first Wednesday'
        subscription = self._create_insight_subscription(interval=1, frequency='weekly', byweekday=['wednesday'], bysetpos=-1)
        assert subscription.summary == 'sent every week on the last Wednesday'
        subscription = self._create_insight_subscription(interval=1, frequency='weekly', byweekday=['wednesday'])
        assert subscription.summary == 'sent every week'
        subscription = self._create_insight_subscription(interval=1, frequency='monthly', byweekday=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], bysetpos=3)
        assert subscription.summary == 'sent every month on the third day'

    def test_subscription_summary_with_unexpected_values(self):
        if False:
            i = 10
            return i + 15
        subscription = self._create_insight_subscription(interval=1, frequency='monthly', byweekday=['monday'], bysetpos=10)
        assert subscription.summary == 'sent on a schedule'