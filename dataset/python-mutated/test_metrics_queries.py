"""
Tests for metrics database queries
"""
from datetime import date, datetime, timezone
from unittest.mock import patch
import pytz
from django.test import RequestFactory
from django.urls import reverse
from dojo.metrics import views
from dojo.models import User
from .dojo_test_case import DojoTestCase

class MockMessages:

    def add(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass

class FindingQueriesTest(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def setUp(self):
        if False:
            while True:
                i = 10
        user = User.objects.get(username='user1')
        self.request = RequestFactory().get(reverse('metrics'))
        self.request.user = user
        self.request._messages = MockMessages()

    def test_finding_queries_no_data(self):
        if False:
            for i in range(10):
                print('nop')
        user3 = User.objects.get(username='user3')
        self.request.user = user3
        product_types = []
        finding_queries = views.finding_querys(product_types, self.request)
        self.assertSequenceEqual(finding_queries['all'].values(), [])

    @patch('django.utils.timezone.now')
    def test_finding_queries(self, mock_timezone):
        if False:
            for i in range(10):
                print('nop')
        mock_datetime = datetime(2020, 12, 9, tzinfo=timezone.utc)
        mock_timezone.return_value = mock_datetime
        with self.assertNumQueries(27):
            product_types = []
            finding_queries = views.finding_querys(product_types, self.request)
            self.assertSequenceEqual(list(finding_queries.keys()), ['all', 'closed', 'accepted', 'accepted_count', 'top_ten', 'monthly_counts', 'weekly_counts', 'weeks_between', 'start_date', 'end_date'])
            self.assertSequenceEqual(finding_queries['all'].qs.values(), [])
            self.assertSequenceEqual(finding_queries['closed'].values(), [])
            self.assertSequenceEqual(finding_queries['accepted'].values(), [])
            self.assertSequenceEqual(list(finding_queries['accepted_count'].values()), [None, None, None, None, None, None])
            self.assertSequenceEqual(finding_queries['top_ten'].values(), [])
            self.assertSequenceEqual(list(finding_queries['monthly_counts'].values()), [[['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1604188800000, datetime(2020, 11, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0], [1606780800000, datetime(2020, 12, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1604188800000, datetime(2020, 11, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1606780800000, datetime(2020, 12, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1604188800000, datetime(2020, 11, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1606780800000, datetime(2020, 12, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]]])
            self.assertDictEqual(finding_queries['weekly_counts'], {'opened_per_period': [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1607299200000, datetime(2020, 12, 7, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0], [1607904000000, datetime(2020, 12, 14, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0], [1608508800000, datetime(2020, 12, 21, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0]], 'accepted_per_period': [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1607299200000, datetime(2020, 12, 7, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1607904000000, datetime(2020, 12, 14, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1608508800000, datetime(2020, 12, 21, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]], 'active_per_period': [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1607299200000, datetime(2020, 12, 7, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1607904000000, datetime(2020, 12, 14, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1608508800000, datetime(2020, 12, 21, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]]})
            self.assertEqual(finding_queries['weeks_between'], 2)
            self.assertIsInstance(finding_queries['start_date'], datetime)
            self.assertIsInstance(finding_queries['end_date'], datetime)

class EndpointQueriesTest(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        user = User.objects.get(username='user1')
        self.request = RequestFactory().get(reverse('metrics'))
        self.request.user = user
        self.request._messages = MockMessages()

    def test_endpoint_queries_no_data(self):
        if False:
            return 10
        user3 = User.objects.get(username='user3')
        self.request.user = user3
        product_types = []
        endpoint_queries = views.endpoint_querys(product_types, self.request)
        self.assertSequenceEqual(endpoint_queries['all'].values(), [])

    def test_endpoint_queries(self):
        if False:
            print('Hello World!')
        with self.assertNumQueries(69):
            product_types = []
            endpoint_queries = views.endpoint_querys(product_types, self.request)
            self.assertSequenceEqual(list(endpoint_queries.keys()), ['all', 'closed', 'accepted', 'accepted_count', 'top_ten', 'monthly_counts', 'weekly_counts', 'weeks_between', 'start_date', 'end_date'])
            self.assertSequenceEqual(endpoint_queries['all'].values(), [{'id': 1, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': False, 'risk_accepted': False, 'endpoint_id': 2, 'finding_id': 2, 'endpoint__product__prod_type__member': False, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}, {'id': 3, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': True, 'out_of_scope': False, 'risk_accepted': False, 'endpoint_id': 5, 'finding_id': 228, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}, {'id': 4, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': True, 'risk_accepted': False, 'endpoint_id': 5, 'finding_id': 229, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}, {'id': 5, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': False, 'risk_accepted': True, 'endpoint_id': 5, 'finding_id': 230, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}, {'id': 7, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': False, 'risk_accepted': False, 'endpoint_id': 7, 'finding_id': 227, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}, {'id': 8, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': False, 'risk_accepted': False, 'endpoint_id': 8, 'finding_id': 231, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}])
            self.assertSequenceEqual(endpoint_queries['closed'].values(), [])
            self.assertSequenceEqual(endpoint_queries['accepted'].values(), [{'id': 5, 'date': date(2020, 7, 1), 'last_modified': datetime(2020, 7, 1, 17, 45, 39, 791907, tzinfo=pytz.UTC), 'mitigated': False, 'mitigated_time': None, 'mitigated_by_id': None, 'false_positive': False, 'out_of_scope': False, 'risk_accepted': True, 'endpoint_id': 5, 'finding_id': 230, 'endpoint__product__prod_type__member': True, 'endpoint__product__member': True, 'endpoint__product__prod_type__authorized_group': False, 'endpoint__product__authorized_group': False}])
            self.assertSequenceEqual(list(endpoint_queries['accepted_count'].values()), [1, 0, 0, 0, 0, 1])
            self.assertSequenceEqual(endpoint_queries['top_ten'].values(), [])
            self.assertSequenceEqual(list(endpoint_queries['monthly_counts'].values()), [[['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1590969600000, datetime(2020, 6, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0], [1593561600000, datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1590969600000, datetime(2020, 6, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1593561600000, datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1590969600000, datetime(2020, 6, 1, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1593561600000, datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1]]])
            self.assertSequenceEqual(list(endpoint_queries['weekly_counts'].values()), [[['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1593388800000, datetime(2020, 6, 29, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1, 0], [1593993600000, datetime(2020, 7, 6, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0], [1594598400000, datetime(2020, 7, 13, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1593388800000, datetime(2020, 6, 29, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1593993600000, datetime(2020, 7, 6, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0], [1594598400000, datetime(2020, 7, 13, 0, 0, tzinfo=timezone.utc), 0, 0, 0, 0, 0]], [['Timestamp', 'Date', 'S0', 'S1', 'S2', 'S3', 'Total', 'Closed'], [1593388800000, datetime(2020, 6, 29, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1], [1593993600000, datetime(2020, 7, 6, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1], [1594598400000, datetime(2020, 7, 13, 0, 0, tzinfo=timezone.utc), 0, 1, 0, 0, 1]]])
            self.assertEqual(endpoint_queries['weeks_between'], 2)
            self.assertEqual(endpoint_queries['start_date'], datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc))
            self.assertEqual(endpoint_queries['end_date'], datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc))