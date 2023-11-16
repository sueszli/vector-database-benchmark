from datetime import datetime, timedelta
from time import sleep
from unittest.mock import patch
from zoneinfo import ZoneInfo
from django.http import HttpRequest
from freezegun import freeze_time
from rest_framework.request import Request
from posthog.caching.calculate_results import CLICKHOUSE_MAX_EXECUTION_TIME
from posthog.caching.insight_caching_state import InsightCachingState
from posthog.caching.insights_api import BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL, should_refresh_insight
from posthog.test.base import BaseTest, ClickhouseTestMixin, _create_insight

class TestShouldRefreshInsight(ClickhouseTestMixin, BaseTest):
    refresh_request: Request

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        django_request = HttpRequest()
        django_request.GET['refresh'] = 'true'
        self.refresh_request = Request(django_request)

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_should_return_true_if_refresh_not_requested(self):
        if False:
            for i in range(10):
                print('nop')
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')) - timedelta(days=1))
        (should_refresh_now_none, refresh_frequency_none) = should_refresh_insight(insight, None, request=Request(HttpRequest()))
        django_request = HttpRequest()
        django_request.GET['refresh'] = 'false'
        (should_refresh_now_false, refresh_frequency_false) = should_refresh_insight(insight, None, request=Request(django_request))
        self.assertEqual(should_refresh_now_none, False)
        self.assertEqual(refresh_frequency_none, BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL)
        self.assertEqual(should_refresh_now_false, False)
        self.assertEqual(refresh_frequency_false, BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL)

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_should_return_true_if_refresh_requested(self):
        if False:
            return 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')) - timedelta(days=1))
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL)

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_should_return_true_if_insight_does_not_have_last_refresh(self):
        if False:
            for i in range(10):
                print('nop')
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$pageview'}], 'interval': 'month'}, {})
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL)

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_shared_insights_can_be_refreshed_less_often(self):
        if False:
            return 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')) - timedelta(days=1))
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request, is_shared=True)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=30))

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_insights_with_hour_intervals_can_be_refreshed_more_often(self):
        if False:
            return 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$pageview'}], 'interval': 'hour'}, {})
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=3))
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request, is_shared=True)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=30))

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_insights_with_ranges_lower_than_7_days_can_be_refreshed_more_often(self):
        if False:
            while True:
                i = 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$pageview'}], 'interval': 'day', 'date_from': '-3d'}, {})
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=3))
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=self.refresh_request, is_shared=True)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=30))

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_dashboard_filters_should_override_insight_filters_when_deciding_on_refresh_time(self):
        if False:
            for i in range(10):
                print('nop')
        (insight, _, dashboard_tile) = _create_insight(self.team, {'events': [{'id': '$pageview'}], 'interval': 'month'}, {'interval': 'hour'})
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, dashboard_tile, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)
        self.assertEqual(refresh_frequency, timedelta(minutes=3))

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_should_return_true_if_was_recently_refreshed(self):
        if False:
            while True:
                i = 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')))
        request = HttpRequest()
        (should_refresh_now, refresh_frequency) = should_refresh_insight(insight, None, request=Request(request))
        self.assertEqual(should_refresh_now, False)
        self.assertEqual(refresh_frequency, BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL)

    @patch('posthog.caching.insights_api.sleep', side_effect=sleep)
    def test_should_return_true_if_refresh_just_about_to_time_out_elsewhere(self, mock_sleep):
        if False:
            print('Hello World!')
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')) - timedelta(days=1), last_refresh_queued_at=datetime.now(tz=ZoneInfo('UTC')) - timedelta(seconds=CLICKHOUSE_MAX_EXECUTION_TIME - 0.5))
        (should_refresh_now, _) = should_refresh_insight(insight, None, request=self.refresh_request)
        mock_sleep.assert_called_once_with(1)
        self.assertEqual(should_refresh_now, True)

    @freeze_time('2012-01-14T03:21:34.000Z')
    def test_should_return_true_if_refresh_timed_out_elsewhere_before(self):
        if False:
            while True:
                i = 10
        (insight, _, _) = _create_insight(self.team, {'events': [{'id': '$autocapture'}], 'interval': 'month'}, {})
        InsightCachingState.objects.filter(team=self.team, insight_id=insight.pk).update(last_refresh=datetime.now(tz=ZoneInfo('UTC')) - timedelta(days=1), last_refresh_queued_at=datetime.now(tz=ZoneInfo('UTC')) - timedelta(seconds=500))
        (should_refresh_now, _) = should_refresh_insight(insight, None, request=self.refresh_request)
        self.assertEqual(should_refresh_now, True)