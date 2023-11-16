from datetime import timedelta
from typing import Any, Dict, List, Union
from unittest import mock
import pytest
from django.urls import reverse
from sentry.snuba.metrics.naming_layer import TransactionMRI
from sentry.testutils.cases import MetricsAPIBaseTestCase
from sentry.testutils.helpers.datetime import freeze_time, iso_format
from sentry.testutils.silo import region_silo_test
pytestmark = pytest.mark.sentry_metrics

@region_silo_test(stable=True)
@freeze_time(MetricsAPIBaseTestCase.MOCK_DATETIME)
class OrganizationEventsTrendsStatsV2EndpointTest(MetricsAPIBaseTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(self.user)
        self.org = self.create_organization(owner=self.user)
        self.project = self.create_project(organization=self.org)
        self.url = reverse('sentry-api-0-organization-events-trends-statsv2', args=[self.org.slug])
        self.store_performance_metric(name=TransactionMRI.DURATION.value, tags={'transaction': 'foo'}, org_id=self.org.id, project_id=self.project.id, value=1, hours_before_now=1)
        self.store_performance_metric(name=TransactionMRI.DURATION.value, tags={'transaction': 'foo'}, org_id=self.org.id, project_id=self.project.id, value=2, hours_before_now=2)
        self.store_performance_metric(name=TransactionMRI.DURATION.value, tags={'transaction': 'foo'}, org_id=self.org.id, project_id=self.project.id, value=2, hours_before_now=2)
        self.features = {'organizations:performance-view': True, 'organizations:performance-new-trends': True}

    @property
    def now(self):
        if False:
            print('Hello World!')
        return MetricsAPIBaseTestCase.MOCK_DATETIME

    def test_no_feature_flag(self):
        if False:
            while True:
                i = 10
        response = self.client.get(self.url, format='json', data={'end': iso_format(self.now - timedelta(minutes=1)), 'start': iso_format(self.now - timedelta(hours=4)), 'field': ['project', 'transaction'], 'query': 'event.type:transaction'})
        assert response.status_code == 404, response.content

    def test_no_project(self):
        if False:
            print('Hello World!')
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now - timedelta(minutes=1)), 'start': iso_format(self.now - timedelta(hours=4)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': ''})
        assert response.status_code == 200, response.content
        assert response.data == []

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_trends(self, mock_detect_breakpoints):
        if False:
            i = 10
            return i + 15
        mock_trends_result = [{'project': self.project.slug, 'transaction': 'foo', 'change': 'regression', 'trend_difference': -15, 'trend_percentage': 0.88}]
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id})
        assert response.status_code == 200, response.content
        events = response.data['events']
        result_stats = response.data['stats']
        assert len(events['data']) == 1
        assert events['data'] == mock_trends_result
        assert len(result_stats) > 0
        assert len(result_stats.get(f'{self.project.slug},foo', [])) > 0

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_no_trends(self, mock_detect_breakpoints):
        if False:
            return 10
        mock_trends_result: List[Union[Dict[str, Any], None]] = []
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id})
        assert response.status_code == 200, response.content
        events = response.data['events']
        result_stats = response.data['stats']
        assert len(events['data']) == 0
        assert events['data'] == []
        assert len(result_stats) == 0

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_transaction_query(self, mock_detect_breakpoints):
        if False:
            while True:
                i = 10
        mock_trends_result: List[Union[Dict[str, Any], None]] = []
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        self.store_performance_metric(name=TransactionMRI.DURATION.value, tags={'transaction': 'bar'}, org_id=self.org.id, project_id=self.project.id, value=2, hours_before_now=2)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction transaction:foo', 'project': self.project.id})
        trends_call_args_data = mock_detect_breakpoints.call_args[0][0]['data']
        assert len(trends_call_args_data.get(f'{self.project.slug},foo')) > 0
        assert len(trends_call_args_data.get(f'{self.project.slug},bar', [])) == 0
        assert response.status_code == 200, response.content

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_trends_p75(self, mock_detect_breakpoints):
        if False:
            i = 10
            return i + 15
        mock_trends_result = [{'project': self.project.slug, 'transaction': 'foo', 'change': 'regression', 'trend_difference': -15, 'trend_percentage': 0.88}]
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id, 'trendFunction': 'p75(transaction.duration)'})
        assert response.status_code == 200, response.content
        events = response.data['events']
        result_stats = response.data['stats']
        assert len(events['data']) == 1
        assert events['data'] == mock_trends_result
        assert len(result_stats) > 0
        assert len(result_stats.get(f'{self.project.slug},foo', [])) > 0

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_trends_p95(self, mock_detect_breakpoints):
        if False:
            return 10
        mock_trends_result = [{'project': self.project.slug, 'transaction': 'foo', 'change': 'regression', 'trend_difference': -15, 'trend_percentage': 0.88}]
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id, 'trendFunction': 'p95(transaction.duration)'})
        assert response.status_code == 200, response.content
        events = response.data['events']
        result_stats = response.data['stats']
        assert len(events['data']) == 1
        assert events['data'] == mock_trends_result
        assert len(result_stats) > 0
        assert len(result_stats.get(f'{self.project.slug},foo', [])) > 0

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_trends_p95_with_project_id(self, mock_detect_breakpoints):
        if False:
            while True:
                i = 10
        mock_trends_result = [{'project': self.project.id, 'transaction': 'foo', 'change': 'regression', 'trend_difference': -15, 'trend_percentage': 0.88}]
        mock_detect_breakpoints.return_value = {'data': mock_trends_result}
        with self.feature({**self.features, 'organizations:performance-trendsv2-dev-only': True}):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id, 'trendFunction': 'p95(transaction.duration)'})
        assert response.status_code == 200, response.content
        events = response.data['events']
        result_stats = response.data['stats']
        assert len(events['data']) == 1
        assert events['data'] == mock_trends_result
        assert len(result_stats) > 0
        assert len(result_stats.get(f'{self.project.id},foo', [])) > 0

    @mock.patch('sentry.api.endpoints.organization_events_trends_v2.detect_breakpoints')
    def test_simple_with_top_events(self, mock_detect_breakpoints):
        if False:
            for i in range(10):
                print('nop')
        self.store_performance_metric(name=TransactionMRI.DURATION.value, tags={'transaction': 'bar'}, org_id=self.org.id, project_id=self.project.id, value=2, hours_before_now=2)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'end': iso_format(self.now), 'start': iso_format(self.now - timedelta(days=1)), 'interval': '1h', 'field': ['project', 'transaction'], 'query': 'event.type:transaction', 'project': self.project.id, 'trendFunction': 'p95(transaction.duration)', 'topEvents': 1})
        assert response.status_code == 200, response.content
        trends_call_args_data = mock_detect_breakpoints.call_args[0][0]['data']
        assert len(trends_call_args_data.get(f'{self.project.slug},foo')) > 0
        assert len(trends_call_args_data.get(f'{self.project.slug},bar', [])) == 0