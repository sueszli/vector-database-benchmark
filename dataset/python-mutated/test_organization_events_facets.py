from datetime import timedelta, timezone
from unittest import mock
from uuid import uuid4
import requests
from django.urls import reverse
from django.utils import timezone as django_timezone
from rest_framework.exceptions import ParseError
from sentry.testutils.cases import APITestCase, SnubaTestCase
from sentry.testutils.helpers.datetime import before_now, iso_format
from sentry.testutils.silo import region_silo_test

@region_silo_test
class OrganizationEventsFacetsEndpointTest(SnubaTestCase, APITestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.min_ago = before_now(minutes=1).replace(microsecond=0)
        self.day_ago = before_now(days=1).replace(microsecond=0)
        self.login_as(user=self.user)
        self.project = self.create_project()
        self.project2 = self.create_project()
        self.url = reverse('sentry-api-0-organization-events-facets', kwargs={'organization_slug': self.project.organization.slug})
        self.min_ago_iso = iso_format(self.min_ago)
        self.features = {'organizations:discover-basic': True, 'organizations:global-views': True}

    def assert_facet(self, response, key, expected):
        if False:
            for i in range(10):
                print('nop')
        actual = None
        for facet in response.data:
            if facet['key'] == key:
                actual = facet
                break
        assert actual is not None, f'Could not find {key} facet in {response.data}'
        assert 'topValues' in actual
        key = lambda row: row['name'] if row['name'] is not None else ''
        assert sorted(expected, key=key) == sorted(actual['topValues'], key=key)

    def test_performance_view_feature(self):
        if False:
            while True:
                i = 10
        self.features.update({'organizations:discover-basic': False, 'organizations:performance-view': True})
        with self.feature(self.features):
            response = self.client.get(self.url, data={'project': self.project.id}, format='json')
        assert response.status_code == 200, response.content

    def test_simple(self):
        if False:
            while True:
                i = 10
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}}, project_id=self.project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'two'}}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 2, 'name': 'one', 'value': 'one'}, {'count': 1, 'name': 'two', 'value': 'two'}]
        self.assert_facet(response, 'number', expected)

    def test_order_by(self):
        if False:
            for i in range(10):
                print('nop')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'alpha': 'one'}, 'environment': 'aaaa'}, project_id=self.project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'beta': 'one'}, 'environment': 'bbbb'}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'charlie': 'two'}, 'environment': 'cccc'}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 200, response.content
        keys = [facet['key'] for facet in response.data]
        assert ['alpha', 'beta', 'charlie', 'environment', 'level', 'project'] == keys

    def test_with_message_query(self):
        if False:
            print('Hello World!')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'how to make fast', 'tags': {'color': 'green'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Delet the Data', 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Data the Delet ', 'tags': {'color': 'yellow'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'query': 'delet'}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 1, 'name': 'yellow', 'value': 'yellow'}, {'count': 1, 'name': 'red', 'value': 'red'}]
        self.assert_facet(response, 'color', expected)

    def test_with_condition(self):
        if False:
            while True:
                i = 10
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'how to make fast', 'tags': {'color': 'green'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Delet the Data', 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Data the Delet ', 'tags': {'color': 'yellow'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'query': 'color:yellow'}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 1, 'name': 'yellow', 'value': 'yellow'}]
        self.assert_facet(response, 'color', expected)

    def test_with_conditional_filter(self):
        if False:
            print('Hello World!')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'how to make fast', 'tags': {'color': 'green'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Delet the Data', 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'message': 'Data the Delet ', 'tags': {'color': 'yellow'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'query': 'color:yellow OR color:red'}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 1, 'name': 'yellow', 'value': 'yellow'}, {'count': 1, 'name': 'red', 'value': 'red'}]
        self.assert_facet(response, 'color', expected)

    def test_start_end(self):
        if False:
            i = 10
            return i + 15
        two_days_ago = self.day_ago - timedelta(days=1)
        hour_ago = self.min_ago - timedelta(hours=1)
        two_hours_ago = hour_ago - timedelta(hours=1)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(two_days_ago), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(hour_ago), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(two_hours_ago), 'tags': {'color': 'red'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(django_timezone.now()), 'tags': {'color': 'red'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'start': iso_format(self.day_ago), 'end': iso_format(self.min_ago)}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 2, 'name': 'red', 'value': 'red'}]
        self.assert_facet(response, 'color', expected)

    def test_excluded_tag(self):
        if False:
            print('Hello World!')
        self.user = self.create_user()
        self.user2 = self.create_user()
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(self.day_ago), 'message': 'very bad', 'tags': {'sentry:user': self.user.email}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(self.day_ago), 'message': 'very bad', 'tags': {'sentry:user': self.user2.email}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': iso_format(self.day_ago), 'message': 'very bad', 'tags': {'sentry:user': self.user2.email}}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [self.project.id]})
        assert response.status_code == 200, response.content
        expected = [{'count': 2, 'name': self.user2.email, 'value': self.user2.email}, {'count': 1, 'name': self.user.email, 'value': self.user.email}]
        self.assert_facet(response, 'user', expected)

    def test_no_projects(self):
        if False:
            while True:
                i = 10
        org = self.create_organization(owner=self.user)
        url = reverse('sentry-api-0-organization-events-facets', kwargs={'organization_slug': org.slug})
        with self.feature('organizations:discover-basic'):
            response = self.client.get(url, format='json')
        assert response.status_code == 200, response.content
        assert response.data == []

    def test_multiple_projects_without_global_view(self):
        if False:
            return 10
        self.store_event(data={'event_id': uuid4().hex}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex}, project_id=self.project2.id)
        with self.feature('organizations:discover-basic'):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 400, response.content
        assert response.data == {'detail': 'You cannot view events from multiple projects.'}

    def test_project_selected(self):
        if False:
            print('Hello World!')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'two'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'project': [self.project.id]}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'name': 'two', 'value': 'two', 'count': 1}]
        self.assert_facet(response, 'number', expected)

    def test_project_filtered(self):
        if False:
            for i in range(10):
                print('nop')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'two'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, {'query': f'project:{self.project.slug}'}, format='json')
        assert response.status_code == 200, response.content
        expected = [{'name': 'two', 'value': 'two', 'count': 1}]
        self.assert_facet(response, 'number', expected)

    def test_project_key(self):
        if False:
            return 10
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'green'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}}, project_id=self.project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'green'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'red'}}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 3, 'name': self.project.slug, 'value': self.project.id}, {'count': 1, 'name': self.project2.slug, 'value': self.project2.id}]
        self.assert_facet(response, 'project', expected)

    def test_project_key_with_project_tag(self):
        if False:
            print('Hello World!')
        self.organization.flags.allow_joinleave = False
        self.organization.save()
        member_user = self.create_user()
        team = self.create_team(members=[member_user])
        private_project1 = self.create_project(organization=self.organization, teams=[team])
        private_project2 = self.create_project(organization=self.organization, teams=[team])
        self.login_as(member_user)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'green', 'project': '%d' % private_project1.id}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one', 'project': '%d' % private_project1.id}}, project_id=private_project1.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'green'}}, project_id=private_project1.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'red'}}, project_id=private_project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'color': 'red'}}, project_id=private_project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 2, 'name': private_project1.slug, 'value': private_project1.id}, {'count': 2, 'name': private_project2.slug, 'value': private_project2.id}]
        self.assert_facet(response, 'project', expected)

    def test_malformed_query(self):
        if False:
            i = 10
            return i + 15
        self.store_event(data={'event_id': uuid4().hex}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex}, project_id=self.project2.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'query': '\n\n\n\n'})
        assert response.status_code == 400, response.content
        assert response.data['detail'].endswith('(column 1). This is commonly caused by unmatched parentheses. Enclose any text in double quotes.')

    @mock.patch('sentry.search.events.builder.discover.raw_snql_query')
    def test_handling_snuba_errors(self, mock_query):
        if False:
            i = 10
            return i + 15
        mock_query.side_effect = ParseError('test')
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 400, response.content

    def test_environment(self):
        if False:
            for i in range(10):
                print('nop')
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}, 'environment': 'staging'}, project_id=self.project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'one'}, 'environment': 'production'}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'number': 'two'}}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
            assert response.status_code == 200, response.content
            expected = [{'count': 1, 'name': 'production', 'value': 'production'}, {'count': 1, 'name': 'staging', 'value': 'staging'}, {'count': 1, 'name': None, 'value': None}]
            self.assert_facet(response, 'environment', expected)
        with self.feature(self.features):
            response = self.client.get(self.url, {'environment': 'staging'}, format='json')
            assert response.status_code == 200, response.content
            expected = [{'count': 1, 'name': 'staging', 'value': 'staging'}]
            self.assert_facet(response, 'environment', expected)
        with self.feature(self.features):
            response = self.client.get(self.url, {'environment': ['staging', 'production']}, format='json')
            assert response.status_code == 200, response.content
            expected = [{'count': 1, 'name': 'production', 'value': 'production'}, {'count': 1, 'name': 'staging', 'value': 'staging'}]
            self.assert_facet(response, 'environment', expected)
        with self.feature(self.features):
            response = self.client.get(self.url, {'environment': ['staging', 'production', '']}, format='json')
            assert response.status_code == 200, response.content
            expected = [{'count': 1, 'name': 'production', 'value': 'production'}, {'count': 1, 'name': 'staging', 'value': 'staging'}, {'count': 1, 'name': None, 'value': None}]
            self.assert_facet(response, 'environment', expected)

    def test_out_of_retention(self):
        if False:
            i = 10
            return i + 15
        with self.options({'system.event-retention-days': 10}):
            with self.feature(self.features):
                response = self.client.get(self.url, format='json', data={'start': iso_format(before_now(days=20)), 'end': iso_format(before_now(days=15))})
        assert response.status_code == 400

    @mock.patch('sentry.utils.snuba.quantize_time')
    def test_quantize_dates(self, mock_quantize):
        if False:
            for i in range(10):
                print('nop')
        mock_quantize.return_value = before_now(days=1).replace(tzinfo=timezone.utc)
        with self.feature('organizations:discover-basic'):
            self.client.get(self.url, format='json', data={'statsPeriod': '1h', 'query': '', 'field': ['id', 'timestamp']})
            self.client.get(self.url, format='json', data={'start': iso_format(before_now(days=20)), 'end': iso_format(before_now(days=15)), 'query': '', 'field': ['id', 'timestamp']})
            assert len(mock_quantize.mock_calls) == 0
            self.client.get(self.url, format='json', data={'field': ['id', 'timestamp'], 'statsPeriod': '90d', 'query': ''})
            assert len(mock_quantize.mock_calls) == 2

    def test_device_class(self):
        if False:
            i = 10
            return i + 15
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'device.class': '1'}}, project_id=self.project2.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'device.class': '2'}}, project_id=self.project.id)
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': {'device.class': '3'}}, project_id=self.project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json')
        assert response.status_code == 200, response.content
        expected = [{'count': 1, 'name': 'high', 'value': 'high'}, {'count': 1, 'name': 'medium', 'value': 'medium'}, {'count': 1, 'name': 'low', 'value': 'low'}]
        self.assert_facet(response, 'device.class', expected)

    def test_with_cursor_parameter(self):
        if False:
            i = 10
            return i + 15
        test_project = self.create_project()
        test_tags = {'a': 'one', 'b': 'two', 'c': 'three', 'd': 'four', 'e': 'five', 'f': 'six', 'g': 'seven', 'h': 'eight', 'i': 'nine', 'j': 'ten', 'k': 'eleven'}
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': test_tags}, project_id=test_project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert links[1]['cursor'] == '0:10:0'
        assert len(response.data) == 10
        for tag_key in list(test_tags.keys())[:10]:
            expected = [{'count': 1, 'name': test_tags[tag_key], 'value': test_tags[tag_key]}]
            self.assert_facet(response, tag_key, expected)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id, 'cursor': '0:10:0'})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'false'
        assert len(response.data) == 2
        expected = [{'count': 1, 'name': 'eleven', 'value': 'eleven'}]
        self.assert_facet(response, 'k', expected)
        expected = [{'count': 1, 'name': 'error', 'value': 'error'}]
        self.assert_facet(response, 'level', expected)

    def test_projects_data_are_injected_on_first_page_with_multiple_projects_selected(self):
        if False:
            for i in range(10):
                print('nop')
        test_project = self.create_project()
        test_project2 = self.create_project()
        test_tags = {'a': 'one', 'b': 'two', 'c': 'three', 'd': 'four', 'e': 'five', 'f': 'six', 'g': 'seven', 'h': 'eight', 'i': 'nine', 'j': 'ten', 'k': 'eleven'}
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': test_tags}, project_id=test_project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [test_project.id, test_project2.id]})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert links[1]['cursor'] == '0:10:0'
        assert len(response.data) == 10
        expected = [{'count': 1, 'name': test_project.slug, 'value': test_project.id}]
        self.assert_facet(response, 'project', expected)
        for tag_key in list(test_tags.keys())[:9]:
            expected = [{'count': 1, 'name': test_tags[tag_key], 'value': test_tags[tag_key]}]
            self.assert_facet(response, tag_key, expected)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [test_project.id, test_project2.id], 'cursor': '0:10:0'})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'false'
        assert len(response.data) == 3
        expected = [{'count': 1, 'name': 'ten', 'value': 'ten'}]
        self.assert_facet(response, 'j', expected)
        expected = [{'count': 1, 'name': 'eleven', 'value': 'eleven'}]
        self.assert_facet(response, 'k', expected)
        expected = [{'count': 1, 'name': 'error', 'value': 'error'}]
        self.assert_facet(response, 'level', expected)

    def test_multiple_pages_with_single_project(self):
        if False:
            print('Hello World!')
        test_project = self.create_project()
        test_tags = {str(i): str(i) for i in range(22)}
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': test_tags}, project_id=test_project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert links[1]['cursor'] == '0:10:0'
        assert len(response.data) == 10
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id, 'cursor': links[1]['cursor']})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert len(response.data) == 10
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id, 'cursor': links[1]['cursor']})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'false'
        assert len(response.data) == 3

    def test_multiple_pages_with_multiple_projects(self):
        if False:
            return 10
        test_project = self.create_project()
        test_project2 = self.create_project()
        test_tags = {str(i): str(i) for i in range(22)}
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': test_tags}, project_id=test_project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [test_project.id, test_project2.id]})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert links[1]['cursor'] == '0:10:0'
        assert len(response.data) == 10
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [test_project.id, test_project2.id], 'cursor': links[1]['cursor']})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'true'
        assert len(response.data) == 10
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': [test_project.id, test_project2.id], 'cursor': links[1]['cursor']})
            links = requests.utils.parse_header_links(response.get('link', '').rstrip('>').replace('>,<', ',<'))
        assert response.status_code == 200, response.content
        assert links[1]['results'] == 'false'
        assert len(response.data) == 4

    def test_get_all_tags(self):
        if False:
            for i in range(10):
                print('nop')
        test_project = self.create_project()
        test_tags = {str(i): str(i) for i in range(22)}
        self.store_event(data={'event_id': uuid4().hex, 'timestamp': self.min_ago_iso, 'tags': test_tags}, project_id=test_project.id)
        with self.feature(self.features):
            response = self.client.get(self.url, format='json', data={'project': test_project.id, 'includeAll': True})
        assert response.status_code == 200, response.content
        assert len(response.data) == 23